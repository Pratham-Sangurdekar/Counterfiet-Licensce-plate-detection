## Counterfeit License Plate Detection (and more)

A simple, end-to-end project to find license plates in traffic images/videos and flag potentially counterfeit plates. It also includes an exploratory notebook for spotting suspicious/reckless driving from video.

This repo is intentionally notebook-first: you can open the notebooks, run cells top-to-bottom, and see results without wiring up a full app.

### What you can do here

- Detect Indian license plates in images/videos using YOLOv8
- Automatically crop detected plates for clean downstream processing
- Classify plates as authentic vs. counterfeit using a CNN (.h5 models included)
- Explore suspicious-driving analysis on sample videos

---

## How it works (bird’s-eye view)

1) Detect license plates
- Model: YOLOv8 (Ultralytics) with a dataset focused on Indian license plates
- Outputs: bounding boxes over plates; crops saved to a folder

2) Crop and prepare
- Crops are saved to `wild/cropped_plates/` for clean inputs to the classifier

3) Classify authenticity
- A simple CNN (Keras/TensorFlow) takes plate crops and predicts if they look counterfeit
- Pretrained weights are provided: `plate_classifier.h5`, `license_plate_classifier.h5`

4) (Optional) Analyze driving behavior
- Try the “suspicious driving” notebook on sample videos (reckless driving examples)

> Note: This is a research/learning project. “Counterfeit” is a visual heuristic from a trained classifier; treat results as advisory, not definitive.

---

## What’s inside

- Notebooks
	- `DETECTIONANDCROPPING.ipynb` / `detection_and_crop.ipynb`: Detect plates and crop them out
	- `classify_plate.ipynb`: Load the CNN and classify plate crops
	- `CNNMODEL.ipynb`: Train/experiment with the CNN plate-classifier
	- `SUSDRIVMODEL.ipynb`: Explore suspicious/reckless driving detection from video
	- `VroomBot.ipynb`: Scratchpad/experiments

- Models & weights
	- `yolov8n.pt`: YOLOv8 nano weights used for detection
	- `plate_classifier.h5`, `license_plate_classifier.h5`: Keras CNN weights for authenticity classification

- Data & training artifacts
	- `wild/`: dataset folder (YOLO format) with `train/`, `valid/`, `test/`, `data.yaml`
	- `wild/cropped_plates/`: generated crops of detected plates
	- `yolo_project/runs/`: YOLO training runs/outputs

- Videos
	- Sample clips like `reckless.mp4`, `reckless1.mp4`, … for behavior experiments
	- `suspicious_driving_ai/video input/`: a couple of input examples

---

## Dataset and licensing

- Dataset: “Indian License Plate Detection” (Roboflow Universe)
	- Link: https://universe.roboflow.com/holi-milan/indian-license-plate-detection-6tmbr
	- Size: 833 images (YOLOv8 format), preprocessed to 640×640
	- License: CC BY 4.0 (see `wild/README.dataset.txt` and `wild/README.roboflow.txt`)

Please respect the dataset license when using or sharing models trained on it.

---

## Setup

You can run everything from notebooks. A typical Python environment with the packages below is enough.

Key dependencies (see `wild/requirements.txt`):
- ultralytics (YOLOv8)
- opencv-python
- tensorflow
- torch, torchvision
- numpy, pandas, matplotlib, seaborn, scikit-learn
- jupyterlab / notebook, ipykernel

Tip for Apple Silicon Macs: if you hit TensorFlow install issues, consider the TensorFlow macOS packages; otherwise run CPU-only.

---

## Quickstart (notebook-first)

1) Detect and crop plates
- Open `DETECTIONANDCROPPING.ipynb` (or `detection_and_crop.ipynb`)
- Run cells to load YOLOv8, run detection on your images/videos, and save crops to `wild/cropped_plates/`

2) Classify authenticity
- Open `classify_plate.ipynb`
- Point it to the crops in `wild/cropped_plates/`
- Load `plate_classifier.h5` (or `license_plate_classifier.h5`) and run predictions

3) (Optional) Try suspicious driving analysis
- Open `SUSDRIVMODEL.ipynb`
- Use the sample videos in the repo (e.g., `reckless.mp4`) or put your own under `suspicious_driving_ai/video input/`

Results appear inline in the notebooks (images, plots, predictions).

---

## Training notes

- YOLOv8 (detection)
	- Dataset config: `wild/data.yaml`
	- Training artifacts/logs: `yolo_project/runs/`

- CNN (authenticity classifier)
	- Use `CNNMODEL.ipynb` to train/iterate on the classifier
	- Saved models: `plate_classifier.h5`, `license_plate_classifier.h5`

Feel free to swap in different backbones, augmentations, or transfer learning strategies to improve accuracy.

---

## Repo map (at a glance)

- `classify_plate.ipynb` — classify plate crops as authentic/counterfeit
- `detection_and_crop.ipynb` / `DETECTIONANDCROPPING.ipynb` — detect license plates and save crops
- `CNNMODEL.ipynb` — build/train the plate classifier
- `SUSDRIVMODEL.ipynb` — experimental suspicious driving analysis
- `VroomBot.ipynb` — misc experiments
- `wild/` — dataset, YOLO config (`data.yaml`), and generated `cropped_plates/`
- `yolo_project/` — YOLO training runs and notes
- `yolov8n.pt` — YOLOv8 weights
- `plate_classifier.h5`, `license_plate_classifier.h5` — CNN weights

---

## Responsible use

This project is for research and education. Real-world license plate analysis can have legal and ethical implications. Ensure you have consent to process data, comply with local laws, and do not use any outputs to take consequential actions without human review.

---

## Acknowledgements

- Roboflow Universe dataset: Indian License Plate Detection (CC BY 4.0)
- Ultralytics YOLOv8
- Open-source contributors across the Python CV ecosystem

---

## Questions or ideas?

If you’re exploring this for a class or a side project and want pointers on extending it (tracking, OCR, better classifiers, model serving), open an issue or start a discussion. Happy experimenting!