import pandas as pd

from pima_diabetes_portfolio.modeling import build_svm_pipeline, build_mlp_pipeline


def sample():
    X = pd.DataFrame({
        "preg":[0,1,2,3,1,4,2,5],
        "plas":[90,140,0,130,85,160,100,150],
        "pres":[70,80,0,76,68,82,72,78],
        "skin":[20,30,0,25,18,35,22,32],
        "insu":[80,120,0,100,70,180,90,150],
        "mass":[25,32,0,29,24,36,27,34],
        "pedi":[.2,.6,.3,.5,.1,.8,.4,.7],
        "age":[22,45,33,40,24,51,30,48],
    })
    y = [0,1,0,1,0,1,0,1]
    return X, y


def test_svm_pipeline_fits_with_missing_zero_handling():
    X,y=sample()
    model=build_svm_pipeline(kernel="linear")
    model.fit(X,y)
    assert len(model.predict(X)) == len(y)


def test_mlp_pipeline_fits_with_missing_zero_handling():
    X,y=sample()
    model=build_mlp_pipeline(hidden_layer_sizes=(4,), max_iter=500, solver="lbfgs")
    model.fit(X,y)
    assert len(model.predict(X)) == len(y)
