import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import json
from shapely.geometry import mapping


def export_prediction_geojson(predictions, confidences, metadata, class_names, output_path="predicted_metadata.geojson"):
    features = []

    # Infer species from filename path if not already present
    if "species" not in metadata.columns:
        metadata["species"] = metadata["filename"].apply(lambda x: x.split("\\")[1])

    # Build class-to-species mapping based on prediction order
    # unique_species = metadata["species"].unique().tolist()
    # class_to_species = {i: species for i, species in enumerate(unique_species)}
    # Use the exact class-to-species mapping from training
    class_to_species = {i: species for i, species in enumerate(class_names)}

    for i in range(len(predictions)):
        pred_class = int(predictions[i].item())
        species_name = class_to_species.get(pred_class, "unknown")
        meta_row = metadata.iloc[i]
        confidence = round(confidences[i].item(), 4)

        features.append({
            "type": "Feature",
            "properties": {
                "filename": meta_row["filename"],
                "predicted_class": pred_class,
                "predicted_species": species_name,
                "confidence": confidence
            },
            "geometry": mapping(meta_row.geometry)
        })

    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": "EPSG:32630"
            }
        },
        "features": features
    }

    with open(output_path, "w") as f:
        json.dump(geojson, f)

    print(f"✅ Vector prediction metadata saved to {output_path}")

"""
def export_prediction_geojson(predictions, metadata, output_path="predicted_metadata.geojson"):
    features = []

    for i in range(len(predictions)):
        pred_class = int(predictions[i].item())
        meta_row = metadata.iloc[i]

        features.append({
            "type": "Feature",
            "properties": {
                "filename": meta_row["filename"],
                "predicted_class": pred_class
            },
            "geometry": mapping(meta_row.geometry)
        })

    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": "EPSG:32630"
            }
        },
        "features": features
    }

    with open(output_path, "w") as f:
        json.dump(geojson, f)

    print(f"✅ Vector prediction metadata saved to {output_path}")"""

"""
def export_georeferenced_predictions(predictions, metadata, output_dir, patch_size, crs="EPSG:32630"):
    for i in range(predictions.shape[0]):
        pred_class = predictions[i].argmax().item()
        meta_row = metadata.iloc[i]
        bounds = meta_row.geometry.bounds  # (xmin, ymin, xmax, ymax)
        filename = os.path.basename(meta_row["filename"]).replace(".png", ".tif")
        output_path = os.path.join(output_dir, filename)

        pred_array = np.full((patch_size[1], patch_size[0]), pred_class, dtype=np.uint8)
        transform = from_bounds(*bounds, patch_size[0], patch_size[1])

        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=patch_size[1],
            width=patch_size[0],
            count=1,
            dtype=pred_array.dtype,
            crs=crs,
            transform=transform,
        ) as dst:
            dst.write(pred_array, 1)"""


def train_transforms(width, height, augmentation):
    if augmentation == "position":
        x = A.Compose(
            [
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
                A.Resize(width, height, p=1.),
                A.Rotate(limit=45, p=0.9),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                ToTensorV2(),
                ]
            )
        print("Position Augmentation")
    elif augmentation == "cutout":
        x = A.Compose(
            [
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
                A.Resize(width, height, p=1.),
                A.Cutout(num_holes=1, max_h_size=12, max_w_size=12, fill_value=0, p=0.5),
                ToTensorV2(),
            ]
        )
        print("Cutout Augmentation")
    else:
        x = A.Compose(
            [
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
                A.Resize(width, height, p=1.),
                ToTensorV2(),
            ]
        )
        print("No Augmentation")
    return x


def val_transforms(width, height):
    return A.Compose(
        [
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
            A.Resize(width, height),
            ToTensorV2(),
        ]
    )


def test_transforms(width, height):
    return A.Compose(
        [
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0),
            A.Resize(width, height),
            ToTensorV2(),
        ]
    )




