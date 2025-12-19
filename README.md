# Project-R (Blender Addon)

Project-R is a Blender addon that helps split an **equirectangular (2:1)** world map into **oblique Hammer** “continent sections”, and later reassemble processed sections back into a global equirectangular output.

This workflow is based on the “Reprojecting Maps” step described by Worldbuilding Pasta in *An Apple Pie From Scratch, Part VIIc* ([link](https://worldbuildingpasta.blogspot.com/2023/03/an-apple-pie-from-scratch-part-viic.html#reprojectingmaps)).

## Projection engine (credit)

Project-R uses the excellent **`projectionpasta`** project by Nikolai Hersfeldt / Mads de Silva for the core projection math ([repo](https://github.com/hersfeldtn/projectionpasta)).
We vendor `projectionpasta.py` (and its license) inside this addon under:

- `blender_addons/project_r/vendor/projectionpasta/`
Project-R does **not** use the `projectionpasta` name for the addon itself; this addon is a separate UI/workflow layer built around that projection engine.

## Install (dev)

- In Blender: **Edit → Preferences → Add-ons → Install…**
  - Zip the `blender_addons/project_r/` folder as `project_r.zip` and install it.

## Workflow (current)

1. **Init Project** (creates folders + `manifest.json`)
2. **Load World Map** (choose the global equirectangular colour map)
   - The addon infers global size and warns if the image is not ~2:1.
3. Select faces on the sphere for a continent and click **Create Section from Selected Faces**
4. Process exported crops externally (e.g. Gaea) and drop outputs into `processed/<section_id>/`
5. Click **Reassemble** to stitch processed section files back into `reassembled/`
## Repository

Remote origin: `https://github.com/Cradoux/project-r.git`

