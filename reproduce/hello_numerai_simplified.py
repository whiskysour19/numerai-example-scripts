# Initialize NumerAPI - the official Python API client for Numerai
from numerapi import NumerAPI
import json
import pandas as pd
import lightgbm as lgb
import cloudpickle
import logging
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Completely silence LightGBM warnings (all categories, all messages)
warnings.filterwarnings('ignore', module='lightgbm')

# Configuration
DATA_VERSION = "v5.0"
FEATURE_SET_SIZE = "small"
MODE = 'test' # or 'normal'

# Initialize API client
napi = NumerAPI()
logging.info("Initialized NumerAPI client.")

# List available datasets and versions
all_datasets = napi.list_datasets()
dataset_versions = list(set(d.split('/')[0] for d in all_datasets))
logging.info("Listed available datasets and versions.")

# Print all files available for download for our version
current_version_files = [f for f in all_datasets if f.startswith(DATA_VERSION)]
logging.info("Fetched files available for download for the current version.")

# Download and load feature metadata
napi.download_dataset(f"{DATA_VERSION}/features.json")
feature_metadata = json.load(open(f"{DATA_VERSION}/features.json"))
logging.info("Downloaded and loaded feature metadata.")

# Display feature set sizes
feature_sets = feature_metadata["feature_sets"]
feature_set = feature_sets[FEATURE_SET_SIZE]
logging.info("Selected feature set size.")

# Download and load training data
napi.download_dataset(f"{DATA_VERSION}/train.parquet")
train = pd.read_parquet(
    f"{DATA_VERSION}/train.parquet",
    columns=["era", "target"] + feature_set
)
logging.info("Downloaded and loaded training data.")

# Downsample to every 4th era to reduce memory usage and speedup model training
if MODE == 'test':
    # Downsample 20x for test mode
    train = train[train["era"].isin(train["era"].unique()[::80])]
    logging.info("TEST MODE: Downsampled training data to every 80th era (20x reduction).")
else:
    train = train[train["era"].isin(train["era"].unique()[::4])]
    logging.info("Downsampled training data to every 4th era.")

# Define model
model = lgb.LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.01,
    max_depth=5,
    num_leaves=2**5-1,
    colsample_bytree=0.1
)
logging.info(f"Defined model parameters for data version: {DATA_VERSION}, feature set size: {FEATURE_SET_SIZE}, model parameters: n_estimators=2000, learning_rate=0.01, max_depth=5, num_leaves=2**5-1, colsample_bytree=0.1.")

# Train model with warnings completely disabled
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.fit(
        train[feature_set],
        train["target"]
    )
logging.info("Trained model on training data.")

# Download and load validation data
napi.download_dataset(f"{DATA_VERSION}/validation.parquet")
validation = pd.read_parquet(
    f"{DATA_VERSION}/validation.parquet",
    columns=["era", "data_type", "target"] + feature_set
)
validation = validation[validation["data_type"] == "validation"]
del validation["data_type"]
logging.info("Downloaded and loaded validation data.")

# Downsample validation data
if MODE == 'test':
    # Downsample 20x for test mode
    validation = validation[validation["era"].isin(validation["era"].unique()[::80])]
    logging.info("TEST MODE: Downsampled validation data to every 80th era (20x reduction).")
else:
    validation = validation[validation["era"].isin(validation["era"].unique()[::4])]
    logging.info("Downsampled validation data to every 4th era.")

# Apply embargo to avoid data leakage
last_train_era = int(train["era"].unique()[-1])
eras_to_embargo = [str(era).zfill(4) for era in [last_train_era + i for i in range(4)]]
validation = validation[~validation["era"].isin(eras_to_embargo)]
logging.info("Applied embargo to validation data.")

# Generate validation predictions
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    validation["prediction"] = model.predict(validation[feature_set])
logging.info("Generated validation predictions.")

# Install and import scoring tools
from numerai_tools.scoring import numerai_corr, correlation_contribution
logging.info("Installed and imported scoring tools.")

# Download and join meta_model for validation
napi.download_dataset(f"v4.3/meta_model.parquet", round_num=842)
validation["meta_model"] = pd.read_parquet(
    f"v4.3/meta_model.parquet"
)["numerai_meta_model"]
logging.info("Downloaded and joined meta_model for validation.")

# Calculate performance metrics
per_era_corr = validation.groupby("era", include_groups=False).apply(
    lambda x: numerai_corr(x[["prediction"]].dropna(), x["target"].dropna())
)
per_era_mmc = validation.dropna().groupby("era", include_groups=False).apply(
    lambda x: correlation_contribution(x[["prediction"]], x["meta_model"], x["target"])
)
logging.info("Calculated performance metrics.")

# Calculate summary statistics
corr_mean = per_era_corr.mean()
corr_std = per_era_corr.std(ddof=0)
corr_sharpe = corr_mean / corr_std
corr_max_drawdown = (per_era_corr.cumsum().expanding(min_periods=1).max() - per_era_corr.cumsum()).max()

mmc_mean = per_era_mmc.mean()
mmc_std = per_era_mmc.std(ddof=0)
mmc_sharpe = mmc_mean / mmc_std
mmc_max_drawdown = (per_era_mmc.cumsum().expanding(min_periods=1).max() - per_era_mmc.cumsum()).max()

# Display performance summary in a simple aligned format
print("\nPerformance Summary:")
print("=" * 50)
print(f"{'Metric':<15}{'CORR':>15}{'MMC':>15}")
print("-" * 50)
print(f"{'Mean':<15}{corr_mean.iloc[0]:>15.6f}{mmc_mean.iloc[0]:>15.6f}")
print(f"{'Std':<15}{corr_std.iloc[0]:>15.6f}{mmc_std.iloc[0]:>15.6f}")
print(f"{'Sharpe':<15}{corr_sharpe.iloc[0]:>15.6f}{mmc_sharpe.iloc[0]:>15.6f}")
print(f"{'Max Drawdown':<15}{corr_max_drawdown.iloc[0]:>15.6f}{mmc_max_drawdown.iloc[0]:>15.6f}")
print("=" * 50)
logging.info("Calculated and displayed performance summary.")

# Download and process live data
napi.download_dataset(f"{DATA_VERSION}/live.parquet")
live_features = pd.read_parquet(f"{DATA_VERSION}/live.parquet", columns=feature_set)
logging.info("Making inference on live data.")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    live_predictions = model.predict(live_features[feature_set])
logging.info("Downloaded and processed live data.")

# Define prediction pipeline function
def predict(live_features: pd.DataFrame) -> pd.DataFrame:
    logging.info("Making inference on provided features.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        live_predictions = model.predict(live_features[feature_set])
    submission = pd.Series(live_predictions, index=live_features.index)
    return submission.to_frame("prediction")
logging.info("Defined prediction pipeline function.")

# Serialize prediction function
p = cloudpickle.dumps(predict)
with open("hello_numerai.pkl", "wb") as f:
    f.write(p)
logging.info("Serialized prediction function.")

# Download file if running in Google Colab
try:
    from google.colab import files
    files.download('hello_numerai.pkl')
    logging.info("Downloaded prediction function file.")
except ImportError:
    logging.info("Not running in Google Colab, skipping file download.")