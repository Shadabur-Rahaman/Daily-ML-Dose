def assess_risk(data_sensitivity, model_opacity, deployment_scale):
    score = data_sensitivity * 0.5 + model_opacity * 0.3 + deployment_scale * 0.2
    return "HIGH" if score > 0.7 else "MEDIUM" if score > 0.4 else "LOW"
