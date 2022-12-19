import pickle
import os


def predict_result(data):
    current_file = os.path.abspath(os.path.dirname(__file__))
    test = data
    transform_filename = os.path.join(
        current_file, "C:/Visual Studio/prediction/loan_match")
    ct = pickle.load(open(transform_filename, 'rb'))
    try:
        test_encoded = ct.transform(test)
    except Exception as ex:
        return {"code": 422, "message": str(ex)}

    regressor_filename = os.path.join(
        current_file, 'C:/Visual Studio/prediction/loan_match')
    regressor = pickle.load(open(regressor_filename, 'rb'))
    prediction = regressor.predict(test_encoded)
    if prediction[0] > 0:
        return {"code": 200, "matched": True, "amount": prediction[0]}
    else:
        return {"code": 200, "matched": False, "amount": prediction[0]}
        # return prediction

