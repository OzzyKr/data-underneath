# data-underneath

## Data setup

The dataset is not tracked in git. Download it and place it in the `dataset/` folder at the repo root so the structure looks like this:

```
data-underneath/
└── dataset/
    ├── bbox_annotation/   # bounding-box annotations (.json), one per image
    ├── bbox_vis/          # visualization previews (.png)
    └── images/            # source log images (.tif)
```

Annotation and image files share the same base name, e.g.:

```
dataset/images/0000000009_d-582_0000020435-0001_log01_fe_1in_md.tif
dataset/bbox_annotation/0000000009_d-582_0000020435-0001_log01_fe_1in_md.json
```

The `example/` folder and `example_notebook.ipynb` are also local-only and ignored by git.
