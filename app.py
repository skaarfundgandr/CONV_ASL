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

# TODO: Add frontend WebUI
@app.get("/")
async def index(req: Request):
    context = {
        "framework": "Robyn",
        "templating_engine": "Jinja2"
    }
    return jinja_template.render_template('index.html.jinja', **context)

# TODO: Complete logic to process uploaded image
@app.post("/prediction")
async def predict(req: Request):
    form = await req.form()
    image_file = form["image"]

    image = img.open(io.BytesIO(image_file.body))

if __name__ == "__main__":
    if os.path.exists('model.pth'):
        model = utils.import_model('model.pth')
    app.start()
