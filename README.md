# FSEQ Splitter & ESPixelStick Uploader  
## F-That-Seq
### Generate per-controller FSEQ files, compress + optional FTP upload, verification, reboot

This tool automatically:

- Reads your `xlights_networks.xml`
- Lets you select which controllers to process
- Splits global `.fseq` files into accurate per-controller slices
- Uploads those files to ESPixelStick controllers (Active FTP)
- Optionally verifies and reboots each controller
- Runs in both a GUI and CLI mode
- Auto-installs its required Python packages (`requests`, `tqdm`)

It is built for large xLights installations where precise, controller-aligned distribution is required.

## ⚠ Required Before Use

This tool **only works on**:

### **Rendered xLights sequences saved as “V2 Uncompressed” FSEQ files**

You must:

1. Render your sequences in xLights  
2. Save/export each as **Version 2 Uncompressed `.fseq`**  
3. Place those files in your input directory

Without proper V2 uncompressed files, the splitter has nothing to process.

## Quick Start — GUI

1. Install Python 3.10+  
2. Run:
   ```
   python FSEQ_Split+Send.py
   ```
3. Choose:
   - Input directory of **v2 uncompressed** `.fseq` files  
   - Your `xlights_networks.xml`  
   - Output directory  
4. Select your controllers  
5. Enable or disable Upload / Verify / Reboot  
6. Click **Run**

## Quick Start — CLI

```
python FSEQ_Split+Send.py --cli
```

Prompts will guide you through controller selection and optional FTP/upload settings.

## Features (Summary)

- Controller auto-detection from networks XML  
- Controller selection UI  
- Accurate per-controller slicing based on global channel offsets  
- FTP upload with delete-before-store overwrite handling  
- Optional verification using `/fseqfilelist`  
- Optional reboot via `/X6`  
- Automatic dependency installation  
- GUI by default; CLI available  

## Project Layout

```
FSEQ_Split+Send.py      # Single-file application
README.md    # Documentation
```
