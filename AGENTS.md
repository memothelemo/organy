# Agent Instructions

You're working on a computer vision project entitled Organy, or its longer academic form, "*Assisted*
*Categorization of Photos from Multiple Photographers Using Visual Features and Capture Metadata*."

This project aims to assist users in sorting possible photographs contributed by multiple photographers
whose all of them have different file names, camera clocks, viewpoints and color grading.

After the processing images, a human can review every suggestion and makes a final editorial choice.

## Environment

This project should expect to work with the following system environment conditions:
- A laptop with RTX 2050 (4 GB of VRAM) and 16 GB of RAM
- Windows 11, Linux and WSL, and MacOS
- Every photo must be processed under one second per photo on CPU at preview resolution.

## Limitations and Constraints

These are not negotiable. If a task seems to need breaking one, **stop and ask**. Do not work on it
with your assumptions.

1.  Use Python + OpenCV (`opencv-contrib-python`) for the entire project, especially in the vision pipeline.
2.  Models must run locally after a one time download. No cloud APIs. No training or fine tuning of neural networks.
3.  CPU is the default. Prefer discrete GPU, whenever possible, but do not fall back from CUDA to CPU silently.
4.  Assume all original photographs are read only. Never modify, rename, move or delete files in photo
    folders. Preserve its original filenames and photographer attribution.
5.  Being in the same stack never changes a photo's event, so stacks are independent of events.
6.  Do not merge events automatically. Let the user merge it manually. Visually similar groups from different
    time sessions may stay separate.
7.  Never label similarity, margin, sharpness or exposure values as "confidence" or "%".
8.  Only support photographs in JPEG and PNG formats.
9.  Both ASCII and non-ASCII paths must be supported, so it will be compatible for both Windows, MacOS
    and Linux (in any distribution).
10. Ask before adding any dependency. This applies especially to AGPL code (such as Ultralytics) and
    anything large.
11. Organy must be treated both as a library and a binary. Keep all logic in library modules away from
    the binary modules (`src/organy` and `src/organy/bin` respectively). Keep `__init__.py` light
    so `organy --help` stays fast.
12. Use OpenCV's class style names (e.g. `cv2.ORB.create()`) instead of the flat underscore aliases
    (e.g. `cv2.ORB_create()`). The aliases come from the C++ bindings and Pylance cannot resolve
    them, so it reports errors in human's IDE.
13. All code must pass Pylance without errors. It is strict about type resolution, so add type
    hints wherever it cannot infer a type on its own. Find ways on how to resolve type resolution
    errors.

## How to work

- Prefer the simplest implementation that fully solves the requested problem.
- Keep changes scoped to the request and clean up what your own changes make obsolete. Do not refactor 
  unrelated code.

## Proposed Architecture

This section shows the proposed architecture of Organy. Take note that this architecture is not final
may be subject to changes as the project progresses.

### Stage 1: Importing Photographs

Raw photographs are imported along with its IDs, attributions, EXIF, and SHA1 for easier identification
and served as a temporary cache. This may be useful for identifying duplicated pictures by comparing
the SHA1 hashes.

### Stage 2: Downscaling Imported Photographs

Since computer memory is limited and photographs are mostly large in size, it is recommended to downscale
all imported photographs; not an intended feature to decrease the quality, but to allow the model easily
process each photograph if the user is running on a low powered discrete GPU.

### Stage 3: Removing Duplicate and/or Burst photographs

Processing duplicate and burst photographs is wasteful, therefore the program must identify possible
duplicated and burst photographs before it is even processed for categorization, metadata checking and
other relevant flows.

### Stage 4: Fun Part

This is where the magic happens. No need to explain.

### Stage 5: Export

Once all processing is done, copy selected originals into a separate directory organized by each
possible clusters. Human review is required to verify whether these pictures are being placed in the
expected order.
