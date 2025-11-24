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

try:
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import queue
    TK_AVAILABLE = True
except Exception:
    TK_AVAILABLE = False


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
# Utility: prompt with default
# ---------------------------------------------------------------------------

def prompt_with_default(prompt: str, default: str) -> str:
    text = input(f"{prompt} (default={default}): ").strip()
    return text if text else default


# ---------------------------------------------------------------------------
# Parse xLights networks XML
# ---------------------------------------------------------------------------

def parse_networks_xml(xml_path: Path) -> list[Controller]:
    """
    Parse the xlights_networks.xml and return a list of Controller objects.

    Expects <Controller> elements with:
      - name
      - IP
      - StartChan
      - NumChans

    Adjust this if your XML schema differs.
    """
    if not xml_path.is_file():
        print(f"[ERROR] Networks XML does not exist: {xml_path}")
        sys.exit(1)

    print(f"Parsing networks XML: {xml_path}")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    controllers: list[Controller] = []

    for ctrl_el in root.findall("Controller"):
        name = ctrl_el.get("Name") or ctrl_el.get("name") or "UNKNOWN"
        ip = ctrl_el.get("IP") or ""
        if not ip:
            print(f"[WARN] Controller {name} has no IP, skipping.")
            continue

        start_str = ctrl_el.get("StartChan") or "1"
        num_str = ctrl_el.get("NumChans") or "0"

        try:
            start_chan = int(start_str)
            num_chans = int(num_str)
        except ValueError:
            print(
                f"[WARN] Controller {name} has invalid StartChan/NumChans, "
                f"skipping. StartChan={start_str}, NumChans={num_str}"
            )
            continue

        if num_chans <= 0:
            print(f"[WARN] Controller {name} has NumChans <= 0, skipping.")
            continue

        end_chan = start_chan + num_chans - 1

        controllers.append(
            Controller(
                name=name,
                ip=ip,
                max_channels=num_chans,
                start_channel=start_chan,
                end_channel=end_chan,
            )
        )

    if not controllers:
        print("[ERROR] No valid controllers found in XML.")
        sys.exit(1)

    print(f"Found {len(controllers)} controllers in XML:")
    for c in controllers:
        print(
            f"  {c.name:20s} IP={c.ip:15s} "
            f"Start={c.start_channel:7d} End={c.end_channel:7d} "
            f"(max {c.max_channels} chans)"
        )

    return controllers


# ---------------------------------------------------------------------------
# FSEQ parsing helpers
# ---------------------------------------------------------------------------

FSEQ_MAGIC = b"FSEQ"


@dataclass
class FseqHeader:
    version: int
    channel_count: int
    frame_count: int
    frame_duration_ms: int
    fixed_header_size: int
    sparse_ranges_offset: int
    flags: int
    step_time: int
    compression_type: int
    use_sparse: bool


def parse_fseq_header(path: Path) -> FseqHeader:
    """
    Parse the FSEQ v1/v2 header. Supports sparse ranges for v2.
    """
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != FSEQ_MAGIC:
            raise ValueError(f"{path} is not an FSEQ file (bad magic)")

        version = f.read(1)[0]
        minor_version = f.read(1)[0]

        # Skip "channel count" (4 bytes, LE)
        channel_count_bytes = f.read(4)
        channel_count = struct.unpack("<I", channel_count_bytes)[0]

        # Frame count (4 bytes LE)
        frame_count_bytes = f.read(4)
        frame_count = struct.unpack("<I", frame_count_bytes)[0]

        # Step time (ms)
        step_time_bytes = f.read(2)
        step_time = struct.unpack("<H", step_time_bytes)[0]

        # Fixed header size (2 bytes)
        hdr_size_bytes = f.read(2)
        fixed_header_size = struct.unpack("<H", hdr_size_bytes)[0]

        # Sparse ranges offset (4 bytes)
        sparse_offset_bytes = f.read(4)
        sparse_ranges_offset = struct.unpack("<I", sparse_offset_bytes)[0]

        # Flags (2 bytes)
        flags_bytes = f.read(2)
        flags = struct.unpack("<H", flags_bytes)[0]

        # Compression type (1 byte)
        compression_type_bytes = f.read(1)
        if not compression_type_bytes:
            compression_type = 0
        else:
            compression_type = compression_type_bytes[0]

        # Frame duration (ms) - FPP tends to interpret step_time anyway
        frame_duration_ms = step_time

        use_sparse = bool(flags & 0x0001)  # Example: bit 0 -> sparse

        return FseqHeader(
            version=version,
            channel_count=channel_count,
            frame_count=frame_count,
            frame_duration_ms=frame_duration_ms,
            fixed_header_size=fixed_header_size,
            sparse_ranges_offset=sparse_ranges_offset,
            flags=flags,
            step_time=step_time,
            compression_type=compression_type,
            use_sparse=use_sparse,
        )


def read_fseq_sparse_ranges(f: "io.BufferedReader", header: FseqHeader):
    """
    Read and parse the sparse range table for FSEQ v2.

    Returns:
        list of (start_channel (1-based), channel_count)
    """
    if header.sparse_ranges_offset == 0:
        return []

    f.seek(header.sparse_ranges_offset)
    num_ranges_bytes = f.read(2)
    if len(num_ranges_bytes) < 2:
        return []

    num_ranges = struct.unpack("<H", num_ranges_bytes)[0]
    ranges = []

    for _ in range(num_ranges):
        buf = f.read(8)
        if len(buf) < 8:
            break
        start_chan, count = struct.unpack("<II", buf)
        ranges.append((start_chan, count))

    return ranges


# ---------------------------------------------------------------------------
# Per-controller sparse file writer
# ---------------------------------------------------------------------------

def compute_controller_sparse_ranges(
    controller: Controller,
    global_sparse_ranges: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    Given global sparse ranges and a controller's global start/end channels,
    compute that controller's sparse ranges (global indexing).
    """
    c_start = controller.start_channel
    c_end = controller.end_channel

    result: list[tuple[int, int]] = []
    for g_start, g_len in global_sparse_ranges:
        g_end = g_start + g_len - 1

        # Range intersection with [c_start, c_end]
        inter_start = max(g_start, c_start)
        inter_end = min(g_end, c_end)
        if inter_start <= inter_end:
            inter_len = inter_end - inter_start + 1
            result.append((inter_start, inter_len))

    return result


def write_sparse_fseq_for_controller(
    input_path: Path,
    controller: Controller,
    header: FseqHeader,
    global_sparse_ranges: list[tuple[int, int]],
    output_path: Path,
):
    """
    Create a sparse FSEQ for this controller from the global file.
    """
    with input_path.open("rb") as f_in:
        global_ranges = global_sparse_ranges
        if not global_ranges and header.use_sparse:
            # Attempt to read from file
            f_in.seek(header.sparse_ranges_offset)
            global_ranges = read_fseq_sparse_ranges(f_in, header)

        if not global_ranges:
            raise ValueError(
                f"{input_path} has no sparse ranges but we are in sparse mode."
            )

        ctrl_ranges = compute_controller_sparse_ranges(controller, global_ranges)
        if not ctrl_ranges:
            print(f"  [WARN] Controller {controller.name} has no channels in {input_path.name}")
            return

        # Build the header for the new sparse file
        with output_path.open("wb") as f_out:
            f_out.write(FSEQ_MAGIC)
            f_out.write(struct.pack("B", header.version))
            f_out.write(struct.pack("B", 0))  # minor
            f_out.write(struct.pack("<I", controller.max_channels))
            f_out.write(struct.pack("<I", header.frame_count))
            f_out.write(struct.pack("<H", header.step_time))
            f_out.write(struct.pack("<H", header.fixed_header_size))
            out_sparse_offset = header.fixed_header_size
            f_out.write(struct.pack("<I", out_sparse_offset))
            out_flags = header.flags | 0x0001
            f_out.write(struct.pack("<H", out_flags))
            f_out.write(struct.pack("B", header.compression_type))

            meta_and_sparse_len = header.fixed_header_size - 4 - 1 - 1 - 4 - 4 - 2 - 1
            if meta_and_sparse_len > 0:
                f_out.write(b"\x00" * meta_and_sparse_len)

            f_out.write(struct.pack("<H", len(ctrl_ranges)))
            for start, length in ctrl_ranges:
                local_start = start - controller.start_channel + 1
                f_out.write(struct.pack("<II", local_start, length))

            frame_data_offset = header.fixed_header_size + 2 + len(ctrl_ranges) * 8

            src_channels_per_frame = header.channel_count
            src_frame_len = src_channels_per_frame
            dst_channels_per_frame = controller.max_channels

            f_in.seek(header.fixed_header_size)
            for frame_idx in range(header.frame_count):
                frame_bytes = f_in.read(src_frame_len)
                if len(frame_bytes) < src_frame_len:
                    frame_bytes += b"\x00" * (src_frame_len - len(frame_bytes))

                dst_frame = bytearray(dst_channels_per_frame)
                for g_start, g_len in global_ranges:
                    g_end = g_start + g_len - 1
                    inter_start = max(g_start, controller.start_channel)
                    inter_end = min(g_end, controller.end_channel)
                    if inter_start <= inter_end:
                        inter_len = inter_end - inter_start + 1
                        src_offset = inter_start - 1
                        dst_offset = inter_start - controller.start_channel
                        dst_frame[dst_offset:dst_offset + inter_len] = frame_bytes[
                            src_offset:src_offset + inter_len
                        ]

                f_out.write(dst_frame)


# ---------------------------------------------------------------------------
# Worker wrapper for a single .fseq file
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

    if not input_path.is_file():
        print(f"[WARN] FSEQ not found: {input_path}")
        return jobs

    try:
        header = parse_fseq_header(input_path)
    except Exception as e:
        print(f"[ERROR] Failed to parse header for {input_path.name}: {e}")
        return jobs

    print(
        f"  [INFO] {input_path.name}: ver={header.version}, "
        f"chan={header.channel_count}, frames={header.frame_count}, "
        f"sparse={'yes' if header.use_sparse else 'no'}"
    )

    sparse_ranges = []
    if header.use_sparse:
        with input_path.open("rb") as f_in:
            sparse_ranges = read_fseq_sparse_ranges(f_in, header)

    for ctrl in controllers:
        if ctrl.end_channel < 1 or ctrl.start_channel > header.channel_count:
            continue

        ctrl_dir = Path(output_root) / ctrl.name
        ctrl_dir.mkdir(parents=True, exist_ok=True)

        out_name = input_path.stem + f"_{ctrl.name}.fseq"
        out_path = ctrl_dir / out_name

        try:
            write_sparse_fseq_for_controller(
                input_path=input_path,
                controller=ctrl,
                header=header,
                global_sparse_ranges=sparse_ranges,
                output_path=out_path,
            )
            jobs.append((ctrl.name, ctrl.ip, str(out_path)))
            print(f"    -> wrote {out_path}")
        except Exception as e:
            print(f"[ERROR] Failed to create file for {ctrl.name}: {e}")

    return jobs


# ---------------------------------------------------------------------------
# FTP upload logic
# ---------------------------------------------------------------------------

def ftp_upload_one_controller(
    controller_name: str,
    ip: str,
    files: list[str],
    username: str,
    password: str,
    remote_dir: str,
    debug: bool = False,
):
    """
    Upload all files for a single controller via FTP, sequentially.
    """
    results = []
    try:
        ftp = FTP()
        ftp.connect(ip, 21, timeout=10)
        ftp.login(username, password)

        if remote_dir and remote_dir not in ("/", "."):
            try:
                ftp.cwd(remote_dir)
            except error_perm:
                if debug:
                    print(f"  [{controller_name}] remote_dir '{remote_dir}' missing, creating...")
                ftp.mkd(remote_dir)
                ftp.cwd(remote_dir)

        for local_path in files:
            fname = os.path.basename(local_path)
            try:
                with open(local_path, "rb") as f:
                    if debug:
                        print(f"  [{controller_name}] STOR {fname}")
                    ftp.storbinary(f"STOR {fname}", f)
                results.append({"filename": fname, "local_path": local_path, "ok": True, "error": None})
            except Exception as e:
                err_msg = str(e)
                print(f"[FTP ERROR] {controller_name}@{ip}: {err_msg}")
                results.append({"filename": fname, "local_path": local_path, "ok": False, "error": err_msg})

        ftp.quit()
    except Exception as e:
        err_msg = f"FTP connection error: {e}"
        print(f"[FTP ERROR] {controller_name}@{ip}: {err_msg}")
        for local_path in files:
            fname = os.path.basename(local_path)
            results.append({"filename": fname, "local_path": local_path, "ok": False, "error": err_msg})

    return {
        "controller_name": controller_name,
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
        print("[INFO] No jobs to upload.")
        return {}

    files_by_controller = defaultdict(list)
    for ctrl_name, ip, local_path in jobs:
        files_by_controller[(ctrl_name, ip)].append(local_path)

    print(f"[FTP] Uploading to {len(files_by_controller)} controllers...")

    controller_reports = {}

    with ThreadPoolExecutor(max_workers=min(len(files_by_controller), 8)) as executor:
        future_map = {}
        for (ctrl_name, ip), file_list in files_by_controller.items():
            future = executor.submit(
                ftp_upload_one_controller,
                controller_name=ctrl_name,
                ip=ip,
                files=file_list,
                username=username,
                password=password,
                remote_dir=remote_dir,
                debug=debug,
            )
            future_map[future] = (ctrl_name, ip)

        for future in as_completed(future_map):
            ctrl_name, ip = future_map[future]
            try:
                report = future.result()
                controller_reports[(ctrl_name, ip)] = report
            except Exception as e:
                print(f"[FTP ERROR] Controller {ctrl_name}@{ip}: {e}")

    return controller_reports


# ---------------------------------------------------------------------------
# HTTP-based verification and reboot (FPP-style endpoints)
# ---------------------------------------------------------------------------

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
        if resp.status_code != 200:
            return 0, len(filenames), [f"HTTP {resp.status_code} from {url}"]

        data = resp.text
        verified = 0
        for fname in filenames:
            if fname in data:
                verified += 1
            else:
                errors.append(f"{fname} missing on {ip}")
        return verified, len(filenames), errors

    except Exception as e:
        return 0, len(filenames), [f"HTTP error contacting {ip}: {e}"]


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
        return False, f"HTTP error contacting {ip}: {e}"


# ---------------------------------------------------------------------------
# Final report printer
# ---------------------------------------------------------------------------

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

        reboot_str = "OK" if reboot_ok else ("FAILED" if reboot_err else "N/A")

        notes = []
        if ver_errors:
            notes.append("; ".join(ver_errors))
        if reboot_err:
            notes.append(f"Reboot error: {reboot_err}")

        print(
            f"{ctrl_name:15s} {ip:15s} "
            f"{ok_uploads:3d}/{total_uploads:<8d} "
            f"{ver_ok:3d}/{ver_total:<9d} "
            f"{reboot_str:10s} "
            f"{' | '.join(notes)}"
        )


# ---------------------------------------------------------------------------
# Optional Tkinter GUI front-end
# ---------------------------------------------------------------------------

if TK_AVAILABLE:

    class QueueWriter:
        """Redirect writes to a queue so the GUI can display them safely."""
        def __init__(self, q: "queue.Queue[str]"):
            self.q = q

        def write(self, msg: str):
            if msg:
                self.q.put(msg)

        def flush(self):
            # Needed for compatibility; nothing to do
            pass


    class FSeqGUI:
        def __init__(self, root: "tk.Tk"):
            self.root = root
            self.root.title("FSEQ Splitter / Uploader")
            self.root.geometry("900x600")

            # For redirecting stdout/stderr into the log window
            import sys as _sys
            self.log_queue: "queue.Queue[str]" = queue.Queue()
            self.old_stdout = _sys.stdout
            self.old_stderr = _sys.stderr
            _sys.stdout = QueueWriter(self.log_queue)
            _sys.stderr = QueueWriter(self.log_queue)

            self.worker_thread = None
            self.running = False

            self._build_ui()
            self._poll_log_queue()

        # ----------------------------- UI ---------------------------------
        def _build_ui(self):
            main = ttk.Frame(self.root, padding=10)
            main.pack(fill=tk.BOTH, expand=True)

            # Paths frame
            paths_frame = ttk.LabelFrame(main, text="Paths", padding=10)
            paths_frame.pack(fill=tk.X, expand=False)

            # Input directory
            from pathlib import Path as _Path
            self.input_dir_var = tk.StringVar(value=str(_Path(".").resolve()))
            self._add_browse_row(
                parent=paths_frame,
                label="Input .fseq directory:",
                var=self.input_dir_var,
                browse_cmd=self._browse_input_dir,
            )

            # Networks XML
            default_xml = r"F:\Lights 2025\xlights_networks.xml"
            self.networks_xml_var = tk.StringVar(value=default_xml)
            self._add_browse_row(
                parent=paths_frame,
                label="Networks XML path:",
                var=self.networks_xml_var,
                browse_cmd=self._browse_networks_xml,
            )

            # Output directory
            self.output_dir_var = tk.StringVar(
                value=str((_Path(".") / "fseq_by_controller").resolve())
            )
            self._add_browse_row(
                parent=paths_frame,
                label="Output directory:",
                var=self.output_dir_var,
                browse_cmd=self._browse_output_dir,
            )

            # FTP / Verification / Reboot frame
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

            # Run / Close buttons
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

            self._toggle_ftp_controls()  # Initialize enabled/disabled state

        def _add_browse_row(self, parent, label, var, browse_cmd):
            # Count existing rows by querying current slaves
            row = 0
            for child in parent.grid_slaves():
                row = max(row, int(child.grid_info().get("row", 0)) + 1)

            ttk.Label(parent, text=label).grid(row=row, column=0, sticky="e")
            entry = ttk.Entry(parent, textvariable=var, width=70)
            entry.grid(row=row, column=1, columnspan=2, sticky="we", padx=(5, 5))
            btn = ttk.Button(parent, text="Browse...", command=browse_cmd)
            btn.grid(row=row, column=3, sticky="w")

            parent.columnconfigure(1, weight=1)

        # -------------------------- Browse handlers ------------------------
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

        # --------------------------- FTP controls --------------------------
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

        # ------------------------------ Run logic --------------------------
        def _on_run(self):
            if self.running:
                messagebox.showinfo("Already running", "The process is already running.")
                return

            from pathlib import Path as _Path
            input_dir = _Path(self.input_dir_var.get()).resolve()
            xml_path = _Path(self.networks_xml_var.get()).resolve()
            output_dir = _Path(self.output_dir_var.get()).resolve()

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

            self._clear_log()
            self._append_log("Starting FSEQ processing...\n")
            self.run_button.configure(state="disabled")
            self.running = True

            cfg = {
                "input_dir": input_dir,
                "xml_path": xml_path,
                "output_dir": output_dir,
                "enable_ftp": self.enable_ftp_var.get(),
                "ftp_user": self.ftp_user_var.get(),
                "ftp_pass": self.ftp_pass_var.get(),
                "ftp_dir": self.ftp_dir_var.get(),
                "ftp_debug": self.ftp_debug_var.get(),
                "do_verify": self.verify_var.get(),
                "do_reboot": self.reboot_var.get(),
            }

            import threading as _threading
            self.worker_thread = _threading.Thread(
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
            """Core logic that mirrors the CLI, but uses GUI-provided values."""
            from pathlib import Path as _Path

            print("=== FSEQ splitter (per controller, sparse v2, FTP upload) ===")

            input_dir = cfg["input_dir"]
            xml_path = cfg["xml_path"]
            output_dir = cfg["output_dir"]

            print(f"Input path: {input_dir}")
            print(f"Networks XML: {xml_path}")
            print(f"Output directory: {output_dir}")

            # Parse networks XML -> controllers
            controllers = parse_networks_xml(_Path(xml_path))

            # Find .fseq files (non-recursive)
            input_files = sorted(
                p for p in _Path(input_dir).iterdir()
                if p.is_file() and p.suffix.lower() == ".fseq"
            )

            if not input_files:
                print(f"[ERROR] No .fseq files found in {input_dir}")
                return

            print(f"Found {len(input_files)} .fseq file(s) in {input_dir}")

            # Generate per-controller files (simple serial loop, no multiprocessing)
            upload_jobs: list[tuple[str, str, str]] = []
            for p in input_files:
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
            """Poll the queue for new log messages and append them to the Text box."""
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

        # ------------------------------- Close -----------------------------
        def _on_close(self):
            import sys as _sys
            if self.running:
                if not messagebox.askyesno(
                    "Process running",
                    "The process is still running.\nDo you really want to exit?"
                ):
                    return
            # Restore stdout / stderr
            _sys.stdout = self.old_stdout
            _sys.stderr = self.old_stderr
            self.root.destroy()


    def run_gui():
        """Launch the Tkinter GUI front-end."""
        root = tk.Tk()
        app = FSeqGUI(root)
        root.protocol("WM_DELETE_WINDOW", app._on_close)
        root.mainloop()

else:
    def run_gui():
        """Fallback when Tkinter is not available: run CLI."""
        print("Tkinter not available; falling back to CLI mode.")
        main()


# ---------------------------------------------------------------------------
# Original CLI main()
# ---------------------------------------------------------------------------

def main() -> None:
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


if __name__ == "__main__":
    # If --cli is passed or Tkinter is unavailable, run in classic CLI mode.
    if "--cli" in sys.argv or not TK_AVAILABLE:
        main()
    else:
        run_gui()
