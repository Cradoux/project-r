# Project-R

A Blender addon for splitting equirectangular world maps into low-distortion Hammer projection sections, processing them externally, and reassembling them back into a global map.

## Why Reprojection Matters

When creating fictional world maps, working directly on an equirectangular (2:1) projection causes severe distortion near the poles. Terrain generation tools like **Gaea** or **Wilbur** work best with low-distortion projections.

Project-R solves this by:
1. Splitting your world map into **oblique Hammer projections** centered on each continent
2. Letting you process each section in your preferred terrain tool
3. Reassembling the processed sections back into a seamless global map

This workflow is based on the excellent guide by **Worldbuilding Pasta**: [An Apple Pie From Scratch, Part VIIc - Reprojecting Maps](https://worldbuildingpasta.blogspot.com/2023/03/an-apple-pie-from-scratch-part-viic.html#reprojectingmaps).

---

## Installation

### Step 1: Download the Addon
Download `project_r.zip` from this repository's `blender_addons/` folder.

### Step 2: Install in Blender
1. Open Blender
2. Go to **Edit → Preferences**
3. Click the **Add-ons** tab (left sidebar)
4. Click **Install...** (top right)
5. Navigate to `project_r.zip` and click **Install Add-on**

### Step 3: Enable the Addon
1. In the Add-ons list, search for "Project-R"
2. Check the checkbox to enable it
3. If prompted to install scipy, click **Install scipy** and restart Blender

### Step 4: Find the Panel
1. In the 3D Viewport, press **N** to open the sidebar
2. Click the **Project-R** tab

---

## Project Setup

### Step 1: Create a Project Folder
Create a folder on your computer for this project. This will contain all maps, sections, and outputs.

### Step 2: Prepare Your Source Maps
Place all your equirectangular maps in the `source/` subfolder:
- `source/colour_map.png` - Your world's color/albedo map
- `source/heightmap.exr` - Height/elevation data
- `source/biome_mask.png` - Any other maps you want processed

**Important:** All maps must be the same dimensions with a 2:1 aspect ratio (e.g., 8192×4096).

### Step 3: Initialize the Project
1. In the Project-R panel, set **Project Root** to your project folder
2. Click **Init Project**
   - This creates the folder structure and `manifest.json`
   - If loading an existing project, it will automatically load the world map

### Step 4: Load the World Map
1. Click **Load World Map**
2. Select your color map (this will be displayed on the sphere for visual reference)
3. A UV sphere will be created with your map applied

---

## Selecting Faces for a Section

### Step 1: Enter Edit Mode
1. Select the sphere in the 3D viewport
2. Press **Tab** to enter Edit Mode

### Step 2: Switch to Face Select
1. Press **3** (number row, not numpad) to switch to Face Select mode
2. Or click the face icon in the header bar

### Step 3: Select Faces with Lasso
1. Press **Ctrl + Right Mouse Button** and drag to lasso select faces
2. Select all faces that cover your continent/region of interest
3. **Tip:** Orbit the view (middle mouse drag) to select faces on curved surfaces

### Why Precision Matters
The faces you select directly determine the **blend mask** used during reassembly:
- Selected faces = full opacity in the final output
- Edges of selection = feathered blend zone
- Being precise prevents overlapping artifacts between sections

### Using Expand/Reduce
- **Expand**: Grows your selection by one ring of faces (adds neighboring faces)
- **Reduce**: Shrinks your selection by one ring (removes boundary faces)

Use these to fine-tune your selection after lasso selecting.

---

## Creating a Section

### Step 1: Name Your Section
In the Project-R panel, enter a descriptive name in **Section Name** (e.g., "northern_continent")

### Step 2: Configure Options
- **Square Crop**: Enable if your processing tool requires square images (e.g., Gaea)
- **Feather (px)**: Edge feather size for blending (default 64px works well)

### Step 3: Create the Section
Click **Create Section from Selected Faces**

### What Gets Created
```
project/
├── sections/
│   └── sec_001_northern_continent/
│       ├── crops/
│       │   ├── colour_map.png      ← Cropped color map
│       │   ├── heightmap.exr       ← Cropped height map
│       │   └── biome_mask.png      ← Cropped biome mask
│       ├── colour_map__hammer_full.png
│       └── effective_mask__crop.png  ← Blend mask for this section
```

### Visual Feedback
- The sphere overlay shows colored circles where sections have been extracted
- Adjust **Overlay Opacity** to see the coverage

---

## Processing Your Sections

### Step 1: Process in External Tools
Open the cropped images from `sections/<section_id>/crops/` in your preferred tool:
- **Gaea** - For terrain erosion and detail
- **Wilbur** - For river generation and erosion
- **Photoshop/GIMP** - For manual painting

### Step 2: Save Processed Files
Save your processed outputs to the `processed/` folder with the **same filename**:
```
project/
├── processed/
│   └── sec_001_northern_continent/
│       ├── colour_map.png      ← Processed color map
│       ├── heightmap.exr       ← Processed height map
│       └── gaea_erosion.png    ← New map from Gaea (any name works!)
```

**Important:** Filenames must be consistent across sections for reassembly. If you create `gaea_erosion.png` for one section, create it for all sections.

---

## Reassembling the Global Map

### Step 1: Validate Processed Files
Click **Validate** to check that all sections have processed files.

### Step 2: Configure Reassembly Options
- **Extend Edge Colors**: Enable to fill empty ocean areas by extending from section edges (recommended)

### Step 3: Reassemble
Click **Reassemble**

### Output
Reassembled maps appear in:
```
project/
├── reassembled/
│   ├── colour_map.png
│   ├── heightmap.exr
│   └── gaea_erosion.png
```

Each file combines all sections that had a matching filename.

---

## Tips for Best Results

### Overlapping Selections
- It's OK for sections to overlap slightly
- The blend mask uses "max wins" logic - the section with stronger mask value takes priority
- Feathering creates smooth transitions

### Consistent Processing
- Apply the same processing steps to all sections for visual consistency
- Match brightness/contrast across sections before reassembly

### Iterative Workflow
- You can re-run Create Section to regenerate crops (overwrites previous)
- Reassemble can be run multiple times as you refine your processing

---

## Folder Structure Reference

```
project/
├── manifest.json           ← Project metadata and section info
├── source/                 ← Your original equirectangular maps
│   ├── colour_map.png
│   └── heightmap.exr
├── sections/               ← Generated section crops
│   ├── sec_001_name/
│   │   ├── crops/          ← Cropped maps for processing
│   │   └── effective_mask__crop.png
│   └── sec_002_name/
│       └── ...
├── processed/              ← Your processed outputs (you create these)
│   ├── sec_001_name/
│   │   └── colour_map.png
│   └── sec_002_name/
│       └── colour_map.png
└── reassembled/            ← Final combined outputs
    └── colour_map.png
```

---

## Credits

### Projection Engine
Project-R uses **projectionpasta** by Nikolai Hersfeldt / Mads de Silva for the core projection mathematics.
- Repository: [github.com/hersfeldtn/projectionpasta](https://github.com/hersfeldtn/projectionpasta)
- Vendored under `blender_addons/project_r/vendor/projectionpasta/`

### Workflow Inspiration
Based on the map reprojection workflow from **Worldbuilding Pasta**:
- [An Apple Pie From Scratch, Part VIIc](https://worldbuildingpasta.blogspot.com/2023/03/an-apple-pie-from-scratch-part-viic.html#reprojectingmaps)

---

## Repository

GitHub: [github.com/Cradoux/project-r](https://github.com/Cradoux/project-r)
