import tarfile
from pathlib import Path
from urllib import request

from numpy import squeeze
from tensorflow.compat import v1 as tf1
from tensorflow.python.platform import gfile

from orangecontrib.imageanalytics.local_embedders.local_embedder import LocalEmbedder, MODELS_DIR


class InceptionV3Embedder(LocalEmbedder):

    embedder = None
    model_checkpoint_filename = "classify_image_graph_def.pb"
    model_url = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"

    def _download_model(self, model_dir, output_file):
        model_tar = model_dir/Path("inception-2015-12-05.tgz")
        request.urlretrieve(url=self.model_url, filename=model_tar)
        with tarfile.open(model_tar, "r:gz") as tar_file:
            tar_file.extract(output_file, model_dir)
        assert (model_dir/Path(output_file)).exists(), "Failed to download file"

    def get_model(self):

        model_dir = Path.home() / MODELS_DIR / Path("inception-v3")
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / Path(self.model_checkpoint_filename)
        if not model_path.exists():
            self._download_model(model_dir, self.model_checkpoint_filename)

        return model_path


    def _load_model(self):
        tf1.disable_v2_behavior()
        model_path = self.get_model()
        with gfile.FastGFile(str(model_path), 'rb') as f:
            graph_def = tf1.GraphDef()
            graph_def.ParseFromString(f.read())
        tf1.import_graph_def(graph_def, name='')
        sess = tf1.Session()
        embed_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        self.embedder = lambda images: squeeze(sess.run(embed_tensor, {'DecodeJpeg/contents:0': images}))

    def _load_image(self, image_path):
        return gfile.FastGFile(image_path, 'rb').read()
