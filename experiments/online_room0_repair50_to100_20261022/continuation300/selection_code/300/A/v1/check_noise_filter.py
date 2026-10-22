import numpy as np
from noise_filter import shape_features,classify

def verdict(a):return classify(shape_features(a))
assert verdict(np.ones((9,11),bool))['reason']=='AREA_LT_100'
assert not verdict(np.ones((10,10),bool))['exclude'], 'Exactly 100 compact pixels must survive'
assert not verdict(np.ones((25,35),bool))['exclude'], 'Small solid socket-shaped mask'
assert not verdict(np.ones((18,82),bool))['exclude'], 'Substantial elongated object must survive'
assert verdict(np.ones((2,160),bool))['reason']=='THIN_EDGE_BAND'
outline=np.ones((240,320),bool);outline[6:-6,6:-6]=False
assert verdict(outline)['exclude'], 'Large thin outline is not rescued by its total area'
scatter=np.zeros((400,600),bool);scatter[::15,::15]=True
assert verdict(scatter)['exclude'], 'Disconnected speckles need not have a long thin bounding box'
a=np.zeros((80,80),bool);a[0:25,0:35]=True
b=np.zeros((80,80),bool);b[20:45,20:55]=True
assert shape_features(a)==shape_features(b), 'Image-border position must not change thickness'
f=dict(area=100,extent=100,bbox_area=100,core_pixels={'2':5,'3':5,'8':5,'12':5})
assert 'THIN_EDGE_BAND' in classify(f)['matched_rules']
f['core_pixels']['3']=6
assert 'THIN_EDGE_BAND' not in classify(f)['matched_rules'], '95% equality is inclusive'
print('NOISE_RULES_PASS')
