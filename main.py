from fastapi import FastAPI, Response, status
from typing import Union

from fastapi import FastAPI
from app import train
from app import test
from pydantic import BaseModel


app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


# @app.post("/items/")
# async def create_item(item: Item):
#     return item
# @app.get("/welcome/{name}")
# def read_item(name):
#     full_msg = demo.welcome_message(name)
#     return {"message": full_msg}


@app.get("/match_making/train")
def read_root():
    train.train_data()
    return {"Hello": "Training Demo"}


class Data(BaseModel):
    lead: str


@app.post("/match_making/test")
def predict_result(data: Data, response: Response):
    pred = test.predict_result([data.lead.split(",")])
    print(pred['code'])
    if pred['code'] == 422:
        response.status_code = 422
    return pred
    # return {"Hello": "Training Demo"}

