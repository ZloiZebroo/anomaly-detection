from pyod.models.abod import ABOD
from pyod.models.alad import ALAD
from pyod.models.anogan import AnoGAN
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.auto_encoder_torch import AutoEncoder as AutoEncoderTorch
from pyod.models.cblof import CBLOF
from pyod.models.cd import CD
from pyod.models.cof import COF
from pyod.models.copod import COPOD
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.ecod import ECOD
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.gmm import GMM
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.inne import INNE
from pyod.models.kde import KDE
from pyod.models.knn import KNN
from pyod.models.kpca import KPCA
from pyod.models.lmdd import LMDD
from pyod.models.loci import LOCI
from pyod.models.loda import LODA
from pyod.models.lof import LOF
from pyod.models.lscp import LSCP
from pyod.models.lunar import LUNAR
from pyod.models.mad import MAD
from pyod.models.mcd import MCD
from pyod.models.mo_gaal import MO_GAAL
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA
from pyod.models.rgraph import RGraph
from pyod.models.rod import ROD
from pyod.models.sampling import Sampling
from pyod.models.so_gaal import SO_GAAL
from pyod.models.sod import SOD
from pyod.models.sos import SOS
from pyod.models.suod import SUOD
from pyod.models.vae import VAE
from pyod.models.xgbod import XGBOD

models = {
    'ABOD': ABOD,
    'AnoGAN': AnoGAN,
    'AutoEncoder': AutoEncoder,
    'CBLOF': CBLOF,
    'COPOD': COPOD,
    'DeepSVDD': DeepSVDD,
    'FeatureBagging': FeatureBagging,
    'GMM': GMM,
    'HBOS': HBOS,
    'IForest': IForest,
    'INNE': INNE,
    'KDE': KDE,
    'KPCA': KPCA,
    'LMDD': LMDD,
    'LOCI': LOCI,
    'LODA': LODA,
    'LOF': LOF,
    'LUNAR': LUNAR,
    'MCD': MCD,
    'OCSVM': OCSVM,
    'RGraph': RGraph,
    'ROD': ROD,
    'SOD': SOD,
    'SOS': SOS,
    'VAE': VAE,
}