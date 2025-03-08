from preporcess import prepare

in_dir = r"C:\Users\rajwn\Videos\monai_ML_project\Atlas\atlas-train-dataset-1.0.1\atlas-train-dataset-1.0.1\train"
train_loader, test_loader = prepare(in_dir)