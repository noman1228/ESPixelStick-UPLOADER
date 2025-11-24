FSeq.py - README (Text Version)
---------------------------------

Per-Controller FSEQ Splitter, Uploader, and Verifier

FSeq.py converts global xLights FSEQ v2 files into per-controller files using
definitions from xlights_networks.xml. It can optionally upload, verify, and
reboot controllers. This provides a fully synchronized set of ESP-based pixel
controllers with minimum effort.

--------FEATURES--------
  - Reads controllers, IPs, and MaxChannels from networks XML.
  - Splits global FSEQ v2 (uncompressed only) into per-controller slices.
  - Writes clean FSEQ v2 files without sparse tables.
  - Organizes output by controller:
      output/<ControllerName>/Show.fseq
  - Optional FTP upload (active mode).
  - Optional HTTP verification.
  - Optional reboot via /X6 endpoint.
  - Final sync summary report.

REQUIREMENTS
------------
Python 3.x

Optional:

    pip install tqdm requests


INPUTS
------
  - Global .fseq files (v2, uncompressed)
  - xlights_networks.xml with valid controllers

USAGE
-----
Run:

    python FSeq.py

Prompts:
    1) Directory containing .fseq files
    2) Path to xlights_networks.xml
    3) Output directory
    4) FTP upload? (y/n)
    5) FTP username/password
    6) FTP remote directory
    7) Verify via HTTP? (y/n)
    8) Reboot controllers? (y/n)

WORKFLOW
--------
  1) Parse xlights_networks.xml
  2) Determine channel ranges per controller
  3) Read each global FSEQ
  4) Slice each frame for each controller
  5) Write per-controller FSEQ files
  6) (Optional) Upload files via FTP
  7) (Optional) Verify via HTTP
  8) (Optional) Reboot controllers
  9) Print final sync summary

NOTES
-----
  - Only FSEQ v2 uncompressed is supported.
  - FTP uses ACTIVE mode.
  - HTTP endpoints required:
      /fseqfilelist  for verification
      /X6            for reboot

LICENSE
-------
  Free to use and modify.
