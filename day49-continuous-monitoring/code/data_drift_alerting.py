from scipy.stats import ks_2samp

def detect_data_drift(feature_baseline, feature_live, alpha=0.05):
    stat, p = ks_2samp(feature_baseline, feature_live)
    if p < alpha:
        print(f"⚠️ Data drift (KS p={p:.3f})")
