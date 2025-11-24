#!/usr/bin/env python
import os
import sys
import struct
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from ftplib import FTP, error_perm
import multiprocessing as mp
from functools import partial
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# ---------------- Tkinter / GUI imports ----------------
try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import queue
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False

# ---------------- Optional libs ----------------
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import requests
except ImportError:
    requests = None


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Controller:
    name: str
    ip: str
    max_channels: int
    start_channel: int  # 1-based global start channel
    end_channel: int    # 1-based global end channel


# ---------------------------------------------------------------------------
# Utility: prompt with default (CLI)
# ---------------------------------------------------------------------------

def prompt_with_default(prompt: str, default: str) -> str:
    text = input(f"{prompt} (default={default}): ").strip()
    return text if text else default


# ---------------------------------------------------------------------------
# Parse xLights networks XML
# ---------------------------------------------------------------------------

def parse_networks_xml(xml_path: Path) -> list[Controller]:
    if not xml_path.is_file():
        print(f"[ERROR] Networks XML not found: {xml_path}")
        sys.exit(1)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    controllers: list[Controller] = []
    next_start_channel = 1  # 1-based running channel index

    for ctrl_el in root.findall("Controller"):
        name = ctrl_el.get("Name") or "UNKNOWN"
        ip = ctrl_el.get("IP") or ""
        net_el = ctrl_el.find("network")
        if net_el is None:
            print(f"[WARN] Controller '{name}' has no <network> element, skipping.")
            continue

        max_ch_str = net_el.get("MaxChannels")
        if not max_ch_str:
            print(f"[WARN] Controller '{name}' has no MaxChannels, skipping.")
            continue

        try:
            max_channels = int(max_ch_str)
        except ValueError:
            print(f"[WARN] Controller '{name}' has invalid MaxChannels '{max_ch_str}', skipping.")
            continue

        start_chan = next_start_channel
        end_chan = start_chan + max_channels - 1
        next_start_channel = end_chan + 1

        controllers.append(
            Controller(
                name=name,
                ip=ip,
                max_channels=max_channels,
                start_channel=start_chan,
                end_channel=end_chan,
            )
        )

    if not controllers:
        print("[ERROR] No valid controllers found in networks XML.")
        sys.exit(1)

    print(f"Found {len(controllers)} controllers:")
    for c in controllers:
        print(
            f"  {c.name:15s} IP={c.ip:15s} "
            f"Channels={c.max_channels:6d} GlobalRange=[{c.start_channel}-{c.end_channel}]"
        )

    return controllers


# ---------------------------------------------------------------------------
# FSEQ parsing helpers (v2, uncompressed)
# ---------------------------------------------------------------------------

class FseqFormatError(Exception):
    pass


def read_fseq_header(f) -> tuple[bytes, bytes, int, int, int, int, int]:
    """
    Read the base 32-byte header and any extra header bytes up to ChannelDataOffset.

    Returns:
        base_header (32 bytes),
        extra_header (variable, may be empty),
        channel_data_offset,
        channel_count_per_frame,
        frame_count,
        step_time_ms,
        compression_type
    """
    base_header = f.read(32)
    if len(base_header) != 32:
        raise FseqFormatError("File too small for FSEQ v2 header (32 bytes).")

    # "FSEQ" / "PSEQ" magic
    magic = base_header[0:4]
    if magic not in (b"FSEQ", b"PSEQ"):
        raise FseqFormatError(f"Unknown FSEQ magic {magic!r}.")

    channel_data_offset = int.from_bytes(base_header[4:6], "little")
    header_size = int.from_bytes(base_header[8:10], "little")
    if channel_data_offset < 32:
        raise FseqFormatError(
            f"Invalid ChannelDataOffset {channel_data_offset}, must be >= 32."
        )

    channel_count_per_frame = int.from_bytes(base_header[10:14], "little")
    frame_count = int.from_bytes(base_header[14:18], "little")
    step_time_ms = base_header[18]

    comp_byte = base_header[20]
    compression_type = comp_byte & 0x0F
    sparse_range_count = base_header[22]

    # For our purposes we only support uncompressed v2 files as input
    if compression_type != 0:
        raise FseqFormatError(
            f"Compression type {compression_type} not supported (only uncompressed)."
        )

    # There may be variable headers etc. between 32 and channel_data_offset
    extra_len = channel_data_offset - 32
    if extra_len < 0:
        extra_len = 0
    extra_header = f.read(extra_len)
    if len(extra_header) != extra_len:
        raise FseqFormatError(
            f"Truncated header: expected {extra_len} extra bytes, got {len(extra_header)}."
        )

    # header_size should generally match channel_data_offset, but we don't rely on it.
    return (
        base_header,
        extra_header,
        channel_data_offset,
        channel_count_per_frame,
        frame_count,
        step_time_ms,
        compression_type,
    )


def build_sparse_header_for_controller(
    base_header: bytes,
    extra_header: bytes,
    controller: Controller,
) -> bytes:
    """
    Build a new FSEQ v2 header for a single controller, **without** sparse ranges.

    We keep the file as:
      - v2 uncompressed
      - per-controller channel count
      - no sparse table (sparse_range_count = 0)

    Since the per-controller file already contains only that controller's
    channels in a dense block, sparse ranges are unnecessary and cause
    ESPS to complain.
    """
    header = bytearray(base_header)

    # No sparse ranges
    sparse_count = 0
    sparse_table_len = 0

    # New header size = base 32 + existing extra_header, no sparse table
    header_size = 32 + len(extra_header) + sparse_table_len

    # Channel Data Offset (bytes 4-5) and Header Size (bytes 8-9)
    header[4:6] = header_size.to_bytes(2, "little")
    header[8:10] = header_size.to_bytes(2, "little")

    # Channel count per frame: this controller only
    header[10:14] = controller.max_channels.to_bytes(4, "little")

    # CompressionType/ExtBlockCount at byte 20:
    # keep upper 4 bits (extended compression block count), force compression type to 0
    header[20] = (header[20] & 0xF0) | 0x00

    # Compression Block Count at byte 21 (lower 8 bits) = 0 (uncompressed)
    header[21] = 0

    # Sparse Range Count at byte 22 = 0 (no sparse data present)
    header[22] = sparse_count

    # Reserved / flags at byte 23: leave as-is or zero if you want
    # header[23] = 0

    # No sparse table appended
    header_bytes = bytes(header) + extra_header
    if len(header_bytes) != header_size:
        raise RuntimeError(
            f"Internal error: header_size={header_size}, actual={len(header_bytes)}"
        )

    return header_bytes


# ---------------------------------------------------------------------------
# Worker: process a single input .fseq into per-controller sparse .fseq files
# ---------------------------------------------------------------------------

def worker_generate_for_file(
    input_path_str: str,
    controllers: list[Controller],
    output_root: str,
) -> list[tuple[str, str, str]]:
    """
    Process one global .fseq file and create per-controller sparse files.

    Returns list of (controller_name, controller_ip, local_output_path)
    for later FTP upload.
    """
    input_path = Path(input_path_str)
    jobs: list[tuple[str, str, str]] = []

    try:
        with input_path.open("rb") as src:
            (
                base_header,
                extra_header,
                _channel_data_offset,
                channel_count_per_frame,
                frame_count,
                _step_time_ms,
                _compression_type,
            ) = read_fseq_header(src)
    except FseqFormatError as e:
        print(f"[WARN] Skipping {input_path.name}: {e}")
        return jobs

    # Pre-open output files for each controller and write their header
    controller_files: list[tuple[Controller, object]] = []

    for ctl in controllers:
        out_dir = Path(output_root) / ctl.name
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / input_path.name
        hdr = build_sparse_header_for_controller(base_header, extra_header, ctl)

        dst = out_path.open("wb")
        dst.write(hdr)

        controller_files.append((ctl, dst))
        jobs.append((ctl.name, ctl.ip, str(out_path)))

    # Stream frames
    frame_size = channel_count_per_frame
    with input_path.open("rb") as src:
        _, _, channel_data_offset, _, _, _, _ = read_fseq_header(src)
        src.seek(channel_data_offset, os.SEEK_SET)

        for _ in range(frame_count):
            frame = src.read(frame_size)
            if len(frame) != frame_size:
                print(
                    f"[WARN] {input_path.name}: truncated frame data, "
                    f"expected {frame_size}, got {len(frame)}"
                )
                break

            for ctl, dst in controller_files:
                start = ctl.start_channel - 1
                end = start + ctl.max_channels
                dst.write(frame[start:end])

    # Close outputs
    for ctl, dst in controller_files:
        dst.close()

    return jobs


# ---------------------------------------------------------------------------
# FTP upload helpers – ACTIVE mode + manual USER/PASS (working version)
# ---------------------------------------------------------------------------

def ftp_upload_file(
    ip: str,
    username: str,
    password: str,
    local_path: str,
    remote_dir: str,
    debug: bool = False,
) -> tuple[bool, str | None]:
    """
    Upload a file via FTP in ACTIVE (PORT) mode, with manual USER/PASS
    to work around SimpleFTPServer's broken reply.
    """
    try:
        print(f"[FTP] Connecting to {ip}:21 for {os.path.basename(local_path)}")
        with FTP() as ftp:
            if debug:
                ftp.set_debuglevel(2)

            ftp.connect(ip, 21, timeout=60)
            print(f"[FTP] Connected to {ip}, forcing manual USER/PASS login as {username!r}")

            # Manual USER/PASS dance, ignoring the bogus 220 after USER
            resp_user = ftp.sendcmd(f"USER {username}")
            if debug:
                print(f"[FTP DEBUG] USER resp: {resp_user!r}")

            try:
                resp_pass = ftp.sendcmd(f"PASS {password}")
                if debug:
                    print(f"[FTP DEBUG] PASS resp: {resp_pass!r}")
            except error_perm as e:
                return False, f"Login failed on {ip}: {e}"

            # ACTIVE mode (SimpleFTPServer expects single connection + PORT)
            ftp.set_pasv(False)
            print(f"[FTP] Using ACTIVE mode (PORT) for {ip}")

            # Handle remote directory (optional)
            if remote_dir and remote_dir not in ("/", ".", ""):
                try:
                    ftp.cwd(remote_dir)
                except error_perm:
                    # Try to create path recursively
                    parts = [p for p in remote_dir.split("/") if p]
                    for p in parts:
                        try:
                            ftp.mkd(p)
                        except error_perm:
                            pass
                        ftp.cwd(p)

            filename = os.path.basename(local_path)
            print(f"[FTP] STOR {filename} -> {ip}")
            with open(local_path, "rb") as f:
                ftp.storbinary(f"STOR {filename}", f, blocksize=16 * 1024)

            try:
                ftp.quit()
            except Exception:
                # Some simple servers close immediately; ignore errors on quit.
                pass

        print(f"[FTP] Upload OK: {ip} ({filename})")
        return True, None

    except Exception as e:
        return False, str(e)


def _upload_job_list_for_controller(
    controller_key: tuple[str, str],
    controller_jobs: list[tuple[str, str, str]],
    username: str,
    password: str,
    remote_dir: str,
    debug: bool,
    progress=None,
    lock=None,
):
    """
    Runs in a single thread per controller:
    uploads that controller's files sequentially, updating a shared progress bar.
    """
    ctrl_name, ip = controller_key
    results = []
    for _, _, local_path in controller_jobs:
        ok, err = ftp_upload_file(
            ip=ip,
            username=username,
            password=password,
            local_path=local_path,
            remote_dir=remote_dir,
            debug=debug,
        )
        filename = os.path.basename(local_path)
        results.append(
            {
                "filename": filename,
                "local_path": local_path,
                "ok": ok,
                "error": err,
            }
        )
        if not ok:
            msg = f"[FTP ERROR] {ctrl_name} ({ip}): {err}"
            if tqdm is not None:
                tqdm.write(msg)
            else:
                print(msg)
        if progress is not None and lock is not None:
            with lock:
                progress.update(1)

    return {
        "controller_name": ctrl_name,
        "ip": ip,
        "results": results,
    }


def ftp_upload_all(
    jobs: list[tuple[str, str, str]],
    username: str,
    password: str,
    remote_dir: str,
    debug: bool = False,
):
    """
    Upload all files:
      - One thread per controller (parallel across controllers)
      - Within each controller, files are uploaded sequentially

    Returns:
        dict[(controller_name, ip)] -> {
            "controller_name": ...,
            "ip": ...,
            "results": [
                {"filename": ..., "local_path": ..., "ok": bool, "error": str|None},
                ...
            ]
        }
    """
    if not jobs:
        print("No files to upload.")
        return {}

    print(f"\nGenerated {len(jobs)} files for upload")

    grouped = defaultdict(list)
    for ctrl_name, ip, local_path in jobs:
        if not ip:
            msg = f"[FTP WARN] {ctrl_name} has no IP; skipping upload for {local_path}"
            if tqdm is not None:
                tqdm.write(msg)
            else:
                print(msg)
            continue
        grouped[(ctrl_name, ip)].append((ctrl_name, ip, local_path))

    if not grouped:
        print("No jobs with valid IPs to upload.")
        return {}

    controller_keys = list(grouped.keys())

    progress = None
    lock = None
    if tqdm is not None:
        progress = tqdm(total=sum(len(v) for v in grouped.values()), desc="Uploading")
        lock = threading.Lock()

    controller_reports = {}

    with ThreadPoolExecutor(max_workers=len(controller_keys)) as executor:
        futures = []
        for key in controller_keys:
            controller_jobs = grouped[key]
            fut = executor.submit(
                _upload_job_list_for_controller,
                key,
                controller_jobs,
                username,
                password,
                remote_dir,
                debug,
                progress,
                lock,
            )
            futures.append(fut)

        for fut in as_completed(futures):
            rep = fut.result()
            key = (rep["controller_name"], rep["ip"])
            controller_reports[key] = rep

    if progress is not None:
        progress.close()

    return controller_reports


# ---------------------------------------------------------------------------
# HTTP verification & reboot helpers
# ---------------------------------------------------------------------------

def _extract_remote_names_from_json(obj):
    """
    Try to pull a set of filenames out of whatever JSON structure the controller returns.
    """
    names = set()

    if isinstance(obj, list):
        if all(isinstance(x, str) for x in obj):
            for x in obj:
                names.add(os.path.basename(x))
        elif all(isinstance(x, dict) for x in obj):
            candidate_keys = ["name", "filename", "file", "FileName", "Name"]
            for entry in obj:
                for k in candidate_keys:
                    if k in entry:
                        names.add(os.path.basename(str(entry[k])))
    elif isinstance(obj, dict):
        candidate_keys = ["files", "filelist", "FileList", "FileNames"]
        for k in candidate_keys:
            if k in obj:
                sub = obj[k]
                names.update(_extract_remote_names_from_json(sub))
    return names


def verify_controller_files(ip: str, filenames: list[str], timeout: float = 5.0):
    """
    Hit http://<ip>/fseqfilelist and verify that the given filenames appear
    (by name) on the controller.

    Returns:
        (verified_count, total_count, [error_strings])
    """
    if requests is None:
        return 0, len(filenames), ["requests module not available; HTTP verify skipped"]

    if not filenames:
        return 0, 0, []

    url = f"http://{ip}/fseqfilelist"
    errors = []
    try:
        resp = requests.get(url, timeout=timeout)
    except Exception as e:
        return 0, len(filenames), [f"HTTP error contacting {url}: {e}"]

    if resp.status_code != 200:
        return 0, len(filenames), [f"HTTP {resp.status_code} from {url}"]

    remote_names = set()
    try:
        data = resp.json()
        remote_names = _extract_remote_names_from_json(data)
    except ValueError:
        # Not JSON? Fallback to dumb substring match on text.
        text = resp.text
        for fn in filenames:
            if fn in text:
                remote_names.add(fn)

    # If we didn't get anything useful, try a second time with raw text
    if not remote_names:
        text = resp.text
        for fn in filenames:
            if fn in text:
                remote_names.add(fn)

    verified = 0
    for fn in filenames:
        if fn in remote_names:
            verified += 1

    if verified < len(filenames):
        missing = [fn for fn in filenames if fn not in remote_names]
        if missing:
            errors.append(f"Missing (by name): {', '.join(missing)}")

    return verified, len(filenames), errors


def reboot_controller(ip: str, timeout: float = 5.0):
    """
    POST http://<ip>/X6 to ask the controller to reboot.
    """
    if requests is None:
        return False, "requests module not available; reboot skipped"

    url = f"http://{ip}/X6"
    try:
        resp = requests.post(url, timeout=timeout)
        if 200 <= resp.status_code < 300:
            return True, None
        return False, f"HTTP {resp.status_code} from {url}"
    except Exception as e:
        return False, f"HTTP error contacting {url}: {e}"


def print_sync_report(controller_reports, verification_results, reboot_results):
    """
    Print a consolidated sync report for all controllers.
    """
    print("\n=== Sync Report ===")
    print(
        f"{'Controller':15s} {'IP':15s} "
        f"{'Up OK/Total':12s} {'Ver OK/Total':13s} {'Reboot':10s} {'Notes'}"
    )
    print("-" * 80)

    for key, rep in sorted(controller_reports.items(), key=lambda x: x[0][0]):
        ctrl_name, ip = key
        results = rep["results"]
        total_uploads = len(results)
        ok_uploads = sum(1 for r in results if r["ok"])

        ver_ok, ver_total, ver_errors = verification_results.get(key, (0, 0, []))
        reboot_ok, reboot_err = reboot_results.get(key, (False, None))

        reboot_str = "OK" if reboot_ok else ("FAILED" if reboot_err else "SKIPPED")

        notes_parts = []
        if any(not r["ok"] for r in results):
            failed_files = [r["filename"] for r in results if not r["ok"]]
            notes_parts.append(f"FTP fail: {', '.join(failed_files)}")
        if ver_errors:
            notes_parts.extend(ver_errors)
        if not verification_results:
            notes_parts.append("verify skipped")
        if reboot_err:
            notes_parts.append(reboot_err)

        notes = " | ".join(notes_parts) if notes_parts else ""

        print(
            f"{ctrl_name:15s} {ip:15s} "
            f"{ok_uploads:2d}/{total_uploads:<9d} "
            f"{ver_ok:2d}/{ver_total:<10d} {reboot_str:10s} {notes}"
        )


# ---------------------------------------------------------------------------
# CLI main (your working logic, preserved)
# ---------------------------------------------------------------------------

def cli_main() -> None:
    print("=== FSEQ splitter (per controller, sparse v2, FTP upload) ===")

    # Input directory with global .fseq exports
    input_dir_str = prompt_with_default("Input path", ".")
    input_dir = Path(input_dir_str).resolve()
    if not input_dir.is_dir():
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        sys.exit(1)

    # xLights networks XML
    default_xml = r"F:\Lights 2025\xlights_networks.xml"
    xml_path_str = prompt_with_default("Networks XML path", default_xml)
    xml_path = Path(xml_path_str).resolve()

    controllers = parse_networks_xml(xml_path)

    # Output root directory (controller subdirs will be created inside)
    output_root = prompt_with_default("Output directory", "fseq_by_controller")
    output_root_path = Path(output_root).resolve()
    output_root_path.mkdir(parents=True, exist_ok=True)

    # Find .fseq files (non-recursive)
    input_files = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() == ".fseq"
    )

    if not input_files:
        print(f"[ERROR] No .fseq files found in {input_dir}")
        sys.exit(1)

    print(f"Found {len(input_files)} .fseq file(s) in {input_dir}")

    # Multiprocessing across input files (generation phase)
    use_mp = (len(input_files) > 1) and (mp.cpu_count() > 1)

    upload_jobs: list[tuple[str, str, str]] = []

    if use_mp:
        worker = partial(
            worker_generate_for_file,
            controllers=controllers,
            output_root=str(output_root_path),
        )
        with mp.Pool() as pool:
            it = pool.imap_unordered(worker, [str(p) for p in input_files])
            if tqdm is not None:
                it = tqdm(it, total=len(input_files), desc="Generating")
            for jobs in it:
                upload_jobs.extend(jobs)
    else:
        if tqdm is not None:
            file_iter = tqdm(input_files, desc="Generating")
        else:
            file_iter = input_files

        for p in file_iter:
            jobs = worker_generate_for_file(
                str(p),
                controllers=controllers,
                output_root=str(output_root_path),
            )
            upload_jobs.extend(jobs)

    # FTP upload
    enable_upload = prompt_with_default(
        "Enable FTP upload at END? (y/n)", "y"
    ).strip().lower().startswith("y")

    controller_reports = {}
    verification_results = {}
    reboot_results = {}

    if enable_upload:
        ftp_user = prompt_with_default("FTP username", "esps")
        ftp_pass = prompt_with_default("FTP password", "esps")
        ftp_dir = prompt_with_default("FTP directory", "/")
        debug_str = prompt_with_default("Enable FTP debug output? (y/n)", "n")
        ftp_debug = debug_str.strip().lower().startswith("y")

        controller_reports = ftp_upload_all(
            jobs=upload_jobs,
            username=ftp_user,
            password=ftp_pass,
            remote_dir=ftp_dir,
            debug=ftp_debug,
        )

        if controller_reports:
            # Verify and reboot are optional, layered on top
            do_verify = prompt_with_default(
                "Verify uploaded files via HTTP /fseqfilelist? (y/n)", "y"
            ).strip().lower().startswith("y")

            do_reboot = prompt_with_default(
                "Reboot controllers after upload via /X6? (y/n)", "n"
            ).strip().lower().startswith("y")

            if do_verify:
                for key, rep in controller_reports.items():
                    ctrl_name, ip = key
                    filenames = [r["filename"] for r in rep["results"] if r["ok"]]
                    ver_ok, ver_total, ver_errors = verify_controller_files(ip, filenames)
                    verification_results[key] = (ver_ok, ver_total, ver_errors)

            if do_reboot:
                for key, rep in controller_reports.items():
                    ctrl_name, ip = key
                    ok, err = reboot_controller(ip)
                    reboot_results[key] = (ok, err)
    else:
        print("FTP upload skipped (per user request).")

    # Final sync report
    if controller_reports:
        print_sync_report(controller_reports, verification_results, reboot_results)

    print("Done.")


# ---------------------------------------------------------------------------
# GUI implementation (Windows-friendly)
# ---------------------------------------------------------------------------

if TK_AVAILABLE:

    class QueueWriter:
        """Redirect stdout/stderr to a Tk text widget via a queue."""
        def __init__(self, q: "queue.Queue[str]"):
            self.q = q

        def write(self, msg: str):
            if msg:
                self.q.put(msg)

        def flush(self):
            pass

    class FSeqGUI:
        def __init__(self, root: "tk.Tk"):
            self.root = root
            self.root.title("FSEQ Splitter / ESPixelStick Uploader")
            self.root.geometry("900x600")

            self.log_queue: "queue.Queue[str]" = queue.Queue()
            self.old_stdout = sys.stdout
            self.old_stderr = sys.stderr
            sys.stdout = QueueWriter(self.log_queue)
            sys.stderr = QueueWriter(self.log_queue)

            self.worker_thread = None
            self.running = False

            self._build_ui()
            self._poll_log_queue()

        # ------------------------- UI layout -------------------------

        def _build_ui(self):
            main = ttk.Frame(self.root, padding=10)
            main.pack(fill=tk.BOTH, expand=True)

            # Paths frame
            paths_frame = ttk.LabelFrame(main, text="Paths", padding=10)
            paths_frame.pack(fill=tk.X, expand=False)

            self.input_dir_var = tk.StringVar(value=str(Path(".").resolve()))
            self._add_browse_row(
                parent=paths_frame,
                label="Input .fseq directory:",
                var=self.input_dir_var,
                browse_cmd=self._browse_input_dir,
            )

            default_xml = r"F:\Lights 2025\xlights_networks.xml"
            self.networks_xml_var = tk.StringVar(value=default_xml)
            self._add_browse_row(
                parent=paths_frame,
                label="Networks XML path:",
                var=self.networks_xml_var,
                browse_cmd=self._browse_networks_xml,
            )

            self.output_dir_var = tk.StringVar(
                value=str((Path(".") / "fseq_by_controller").resolve())
            )
            self._add_browse_row(
                parent=paths_frame,
                label="Output directory:",
                var=self.output_dir_var,
                browse_cmd=self._browse_output_dir,
            )

            # FTP / verify / reboot frame
            ftp_frame = ttk.LabelFrame(main, text="FTP / Verification / Reboot", padding=10)
            ftp_frame.pack(fill=tk.X, expand=False, pady=(10, 5))

            self.enable_ftp_var = tk.BooleanVar(value=True)
            self.verify_var = tk.BooleanVar(value=True)
            self.reboot_var = tk.BooleanVar(value=False)
            self.ftp_debug_var = tk.BooleanVar(value=False)

            enable_ftp_cb = ttk.Checkbutton(
                ftp_frame,
                text="Enable FTP upload at END",
                variable=self.enable_ftp_var,
                command=self._toggle_ftp_controls,
            )
            enable_ftp_cb.grid(row=0, column=0, sticky="w")

            self.ftp_user_var = tk.StringVar(value="esps")
            self.ftp_pass_var = tk.StringVar(value="esps")
            self.ftp_dir_var = tk.StringVar(value="/")

            ttk.Label(ftp_frame, text="FTP username:").grid(row=1, column=0, sticky="e")
            self.ftp_user_entry = ttk.Entry(ftp_frame, textvariable=self.ftp_user_var, width=20)
            self.ftp_user_entry.grid(row=1, column=1, sticky="w", padx=(5, 20))

            ttk.Label(ftp_frame, text="FTP password:").grid(row=1, column=2, sticky="e")
            self.ftp_pass_entry = ttk.Entry(
                ftp_frame, textvariable=self.ftp_pass_var, width=20, show="*"
            )
            self.ftp_pass_entry.grid(row=1, column=3, sticky="w", padx=(5, 20))

            ttk.Label(ftp_frame, text="FTP directory:").grid(row=2, column=0, sticky="e")
            self.ftp_dir_entry = ttk.Entry(ftp_frame, textvariable=self.ftp_dir_var, width=20)
            self.ftp_dir_entry.grid(row=2, column=1, sticky="w", padx=(5, 20))

            self.ftp_debug_cb = ttk.Checkbutton(
                ftp_frame, text="Enable FTP debug output", variable=self.ftp_debug_var
            )
            self.ftp_debug_cb.grid(row=2, column=2, sticky="w", pady=(0, 5))

            self.verify_cb = ttk.Checkbutton(
                ftp_frame, text="Verify uploaded files via HTTP /fseqfilelist",
                variable=self.verify_var
            )
            self.verify_cb.grid(row=3, column=0, columnspan=2, sticky="w")

            self.reboot_cb = ttk.Checkbutton(
                ftp_frame, text="Reboot controllers after upload via /X6",
                variable=self.reboot_var
            )
            self.reboot_cb.grid(row=3, column=2, columnspan=2, sticky="w")

            # Buttons
            button_frame = ttk.Frame(main)
            button_frame.pack(fill=tk.X, expand=False, pady=(10, 5))

            self.run_button = ttk.Button(button_frame, text="Run", command=self._on_run)
            self.run_button.pack(side=tk.LEFT)

            self.close_button = ttk.Button(
                button_frame, text="Close", command=self._on_close
            )
            self.close_button.pack(side=tk.LEFT, padx=(10, 0))

            # Log frame
            log_frame = ttk.LabelFrame(main, text="Log", padding=5)
            log_frame.pack(fill=tk.BOTH, expand=True, pady=(5, 0))

            self.log_text = tk.Text(
                log_frame,
                wrap="word",
                height=10,
                state="disabled",
            )
            self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            log_scroll = ttk.Scrollbar(
                log_frame,
                command=self.log_text.yview,
                orient=tk.VERTICAL
            )
            log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            self.log_text["yscrollcommand"] = log_scroll.set

            self._toggle_ftp_controls()

        def _add_browse_row(self, parent, label, var, browse_cmd):
            row = 0
            for child in parent.grid_slaves():
                row = max(row, int(child.grid_info().get("row", 0)) + 1)

            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="e")
            entry = ttk.Entry(parent, textvariable=var, width=70)
            entry.grid(row=row, column=1, columnspan=2, sticky="we", padx=(5, 5))
            btn = ttk.Button(parent, text="Browse...", command=browse_cmd)
            btn.grid(row=row, column=3, sticky="w")

            parent.columnconfigure(1, weight=1)

        def _browse_input_dir(self):
            path = filedialog.askdirectory(title="Select input .fseq directory")
            if path:
                self.input_dir_var.set(path)

        def _browse_networks_xml(self):
            path = filedialog.askopenfilename(
                title="Select xLights networks XML",
                filetypes=[("XML files", "*.xml"), ("All files", "*.*")],
            )
            if path:
                self.networks_xml_var.set(path)

        def _browse_output_dir(self):
            path = filedialog.askdirectory(title="Select output directory")
            if path:
                self.output_dir_var.set(path)

        def _toggle_ftp_controls(self):
            enabled = self.enable_ftp_var.get()
            state = "normal" if enabled else "disabled"
            for widget in [
                self.ftp_user_entry,
                self.ftp_pass_entry,
                self.ftp_dir_entry,
                self.ftp_debug_cb,
                self.verify_cb,
                self.reboot_cb,
            ]:
                widget.configure(state=state)

        # -------- controller selection dialog (from networks XML) ----------
        def _choose_controllers_dialog(self, controllers: list[Controller]):
            win = tk.Toplevel(self.root)
            win.title("Select Controllers")
            win.transient(self.root)
            win.grab_set()

            frm = ttk.Frame(win, padding=10)
            frm.pack(fill=tk.BOTH, expand=True)

            ttk.Label(frm, text="Select controllers to parse/upload:").pack(anchor="w")

            canvas = tk.Canvas(frm, borderwidth=0)
            inner = ttk.Frame(canvas)
            scroll = ttk.Scrollbar(frm, orient="vertical", command=canvas.yview)
            canvas.configure(yscrollcommand=scroll.set)

            scroll.pack(side="right", fill="y")
            canvas.pack(side="left", fill="both", expand=True)
            canvas.create_window((0, 0), window=inner, anchor="nw")

            vars_list = []
            for idx, c in enumerate(controllers):
                var = tk.BooleanVar(value=True)
                txt = f"{c.name} ({c.ip})  Start={c.start_channel}  End={c.end_channel}"
                chk = ttk.Checkbutton(inner, text=txt, variable=var)
                chk.grid(row=idx, column=0, sticky="w")
                vars_list.append(var)

            def _on_configure(event):
                canvas.configure(scrollregion=canvas.bbox("all"))

            inner.bind("<Configure>", _on_configure)

            btn_frame = ttk.Frame(frm)
            btn_frame.pack(fill=tk.X, pady=(8, 0))

            selected_controllers = []
            cancelled = {"flag": False}

            def select_all():
                for v in vars_list:
                    v.set(True)

            def clear_all():
                for v in vars_list:
                    v.set(False)

            def on_ok():
                selected_controllers.clear()
                for v, c in zip(vars_list, controllers):
                    if v.get():
                        selected_controllers.append(c)
                if not selected_controllers:
                    messagebox.showerror(
                        "No controllers selected",
                        "You must select at least one controller.",
                        parent=win,
                    )
                    return
                win.destroy()

            def on_cancel():
                cancelled["flag"] = True
                win.destroy()

            ttk.Button(btn_frame, text="Select All", command=select_all).pack(side=tk.LEFT)
            ttk.Button(btn_frame, text="Clear All", command=clear_all).pack(side=tk.LEFT, padx=(5, 0))
            ttk.Button(btn_frame, text="OK", command=on_ok).pack(side=tk.RIGHT)
            ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side=tk.RIGHT, padx=(5, 0))

            win.wait_window()
            if cancelled["flag"]:
                return None
            return selected_controllers

        # ------------------------------ Run logic --------------------------
        def _on_run(self):
            if self.running:
                messagebox.showinfo("Already running", "The process is already running.")
                return

            input_dir = Path(self.input_dir_var.get()).resolve()
            xml_path = Path(self.networks_xml_var.get()).resolve()
            output_dir = Path(self.output_dir_var.get()).resolve()

            if not input_dir.is_dir():
                messagebox.showerror("Invalid input directory", str(input_dir))
                return
            if not xml_path.is_file():
                messagebox.showerror("Invalid networks XML", str(xml_path))
                return

            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                messagebox.showerror("Output directory error", f"{output_dir}\n{e}")
                return

            # parse networks XML (catch SystemExit from CLI-style function)
            try:
                controllers = parse_networks_xml(xml_path)
            except SystemExit:
                messagebox.showerror("Error", "No valid controllers found in XML.")
                return

            selected_controllers = self._choose_controllers_dialog(controllers)
            if selected_controllers is None:
                return  # cancelled

            self._clear_log()
            self._append_log("Starting FSEQ processing...\n")
            self.run_button.configure(state="disabled")
            self.running = True

            cfg = {
                "input_dir": input_dir,
                "output_dir": output_dir,
                "controllers": selected_controllers,
                "enable_ftp": self.enable_ftp_var.get(),
                "ftp_user": self.ftp_user_var.get(),
                "ftp_pass": self.ftp_pass_var.get(),
                "ftp_dir": self.ftp_dir_var.get(),
                "ftp_debug": self.ftp_debug_var.get(),
                "do_verify": self.verify_var.get(),
                "do_reboot": self.reboot_var.get(),
            }

            self.worker_thread = threading.Thread(
                target=self._run_worker, args=(cfg,), daemon=True
            )
            self.worker_thread.start()

        def _run_worker(self, cfg: dict):
            try:
                self._run_core(cfg)
            except Exception as e:
                print(f"[GUI ERROR] Unexpected error: {e}")
            finally:
                def _done():
                    self.running = False
                    self.run_button.configure(state="normal")
                    self._append_log("\nProcess finished.\n")
                self.root.after(0, _done)

        def _run_core(self, cfg: dict):
            print("=== FSEQ splitter (per controller, sparse v2, FTP upload) ===")

            input_dir = cfg["input_dir"]
            output_dir = cfg["output_dir"]
            controllers = cfg["controllers"]

            print(f"Input path: {input_dir}")
            print(f"Output directory: {output_dir}\n")

            print("Using controllers:")
            for c in controllers:
                print(
                    f"  {c.name:15s} IP={c.ip:15s} "
                    f"Start={c.start_channel:7d} End={c.end_channel:7d}"
                )

            # Find .fseq files (non-recursive)
            input_files = sorted(
                p for p in Path(input_dir).iterdir()
                if p.is_file() and p.suffix.lower() == ".fseq"
            )

            if not input_files:
                print(f"[ERROR] No .fseq files found in {input_dir}")
                return

            print(f"Found {len(input_files)} .fseq file(s) in {input_dir}")

            upload_jobs: list[tuple[str, str, str]] = []

            # For GUI, keep it simple: no multiprocessing, just loop
            file_iter = tqdm(input_files, desc="Generating") if tqdm is not None else input_files
            for p in file_iter:
                print(f"[INFO] Processing {p.name}...")
                jobs = worker_generate_for_file(
                    str(p),
                    controllers=controllers,
                    output_root=str(output_dir),
                )
                upload_jobs.extend(jobs)

            controller_reports = {}
            verification_results = {}
            reboot_results = {}

            if cfg["enable_ftp"]:
                print("\n[FTP] Upload phase enabled.")
                controller_reports = ftp_upload_all(
                    jobs=upload_jobs,
                    username=cfg["ftp_user"],
                    password=cfg["ftp_pass"],
                    remote_dir=cfg["ftp_dir"],
                    debug=cfg["ftp_debug"],
                )

                if controller_reports:
                    if cfg["do_verify"]:
                        print("\n[VERIFY] Verifying uploaded files via HTTP /fseqfilelist")
                        for key, rep in controller_reports.items():
                            ctrl_name, ip = key
                            filenames = [r["filename"] for r in rep["results"] if r["ok"]]
                            ver_ok, ver_total, ver_errors = verify_controller_files(
                                ip, filenames
                            )
                            verification_results[key] = (ver_ok, ver_total, ver_errors)

                    if cfg["do_reboot"]:
                        print("\n[REBOOT] Rebooting controllers via /X6")
                        for key, rep in controller_reports.items():
                            ctrl_name, ip = key
                            ok, err = reboot_controller(ip)
                            reboot_results[key] = (ok, err)
            else:
                print("FTP upload skipped (per GUI setting).")

            if controller_reports:
                print_sync_report(controller_reports, verification_results, reboot_results)

            print("Done.")

        # ------------------------------ Logging ----------------------------
        def _poll_log_queue(self):
            try:
                while True:
                    msg = self.log_queue.get_nowait()
                    self._append_log(msg)
            except queue.Empty:
                pass
            self.root.after(100, self._poll_log_queue)

        def _append_log(self, msg: str):
            self.log_text.configure(state="normal")
            self.log_text.insert(tk.END, msg)
            self.log_text.see(tk.END)
            self.log_text.configure(state="disabled")

        def _clear_log(self):
            self.log_text.configure(state="normal")
            self.log_text.delete("1.0", tk.END)
            self.log_text.configure(state="disabled")

        def _on_close(self):
            if self.running:
                if not messagebox.askyesno(
                    "Process running",
                    "The process is still running.\nDo you really want to exit?"
                ):
                    return
            sys.stdout = self.old_stdout
            sys.stderr = self.old_stderr
            self.root.destroy()


    def run_gui():
        root = tk.Tk()
        app = FSeqGUI(root)
        root.protocol("WM_DELETE_WINDOW", app._on_close)
        root.mainloop()

else:
    def run_gui():
        print("Tkinter not available; falling back to CLI.")
        cli_main()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Force CLI if requested, or if Tkinter unavailable
    if "--cli" in sys.argv or not TK_AVAILABLE:
        cli_main()
    else:
        run_gui()
