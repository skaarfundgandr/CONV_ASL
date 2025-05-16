from robyn import Robyn, Request
from robyn.templating import JinjaTemplate
import utils
import os.path
import pathlib
import PIL.Image as img
import io

app = Robyn(__file__)
current_path = pathlib.Path(__file__).parent.resolve()
jinja_template = JinjaTemplate(os.path.join(current_path, 'templates'))
if os.path.exists('model.pth'):
    model = utils.import_model('model.pth')
else:
    model = utils.train_model()

@app.get("/")
async def index(req: Request):
    return jinja_template.render_template('index.html.jinja')

@app.post("/prediction")
async def predict(req: Request):
    try:
        image = img.open(io.BytesIO(bytearray(req.body)))
        res = utils.predict_from_image(model, image)
        context = {"result": res}
    except Exception as e:
        raise e
    
    return jinja_template.render_template('index.html.jinja', **context)

if __name__ == "__main__":
    app.start(port=5090)
