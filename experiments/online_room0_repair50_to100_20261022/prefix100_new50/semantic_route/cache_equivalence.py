"""Reuse only exact feature vectors after full model/preprocess/config equivalence."""
from pathlib import Path
import hashlib,inspect,json,re,time,importlib
A=Path("/home/chenkejun/CVPR/experiments/online_room0_repair50_to100_20261022/prefix100/semantic_route")
AR=Path("/data/chenkejun/CVPR/results/online_room0_repair50_to100_20261022/prefix100/semantic_route")
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def normalize_model(model):
    out=dict(model);out["preprocess"],n=re.subn(r"(<function _convert_to_rgb) at 0x[0-9a-fA-F]+(>)",r"\1\2",out["preprocess"]);assert n==1
    return out
def exact_namespace(current,config,computed,root):
    other=json.loads((AR/"semantic.json").read_text())
    original=dict(model=other["model"],gray=config["gray"],masked_bbox_square=True,terms=other["terms"],prompts=other["prompts"])
    assert hashlib.sha256(json.dumps(original,sort_keys=True).encode()).hexdigest()==other["cache_namespace"]
    normalized_current=dict(current,model=normalize_model(current["model"]))
    normalized_original=dict(original,model=normalize_model(original["model"]))
    assert normalized_current==normalized_original,"semantic model/preprocessing/terms mismatch: cross-arm reuse forbidden"
    assert sha(A/"config.json")==sha(Path(__file__).parent/"config.json"),"configuration changed"
    transform=importlib.import_module("open_clip.transform")
    code=Path(inspect.getsourcefile(transform._convert_to_rgb))
    assert code.stat().st_mtime < (AR/"semantic.json").stat().st_mtime,"preprocess code newer than source cache"
    proof=dict(status="PASS",namespace_used=other["cache_namespace"],runtime_computed_namespace=computed,
      permitted_normalization="Remove only the detected _convert_to_rgb function repr address from preprocess comparison",
      full_model_meta=normalized_current["model"],gray=config["gray"],masked_bbox_square=True,
      terms=current["terms"],prompts=current["prompts"],config_sha256=sha(A/"config.json"),
      openclip_preprocess_code=str(code),openclip_preprocess_code_sha256=sha(code),
      openclip_preprocess_unchanged_since_source_cache=True,
      inference_code_equivalence_before_cache_adapter=json.loads((root.parent/"exact_semantic_cache_audit.json").read_text())["model_code_equal_after_output_path_normalization"],
      input_key="SHA256(equivalent model namespace + square.shape + exact square RGB bytes)",
      semantic_decisions_reused=False,at=time.time())
    (root/"cache_equivalence.json").write_text(json.dumps(proof,ensure_ascii=False,indent=2))
    return other["cache_namespace"]

class FeatureVerifier:
    def __init__(self,model,preprocess,text_features,threshold):
        self.model=model;self.preprocess=preprocess;self.text=text_features.cpu().numpy()
        self.threshold=threshold;self.closest={};self.first={};self.seen=set()
    def consider(self,key,square,vector,n_target):
        import numpy as np
        if key in self.seen:return
        self.seen.add(key)
        scores=vector@self.text.T
        margin=float(np.max(scores[:n_target])-np.max(scores[n_target:]))
        distance=abs(margin-self.threshold)
        close=len(self.closest)<8 or distance<max(v[0] for v in self.closest.values())
        first=len(self.first)<8 or key<max(self.first)
        if not close and not first:return
        from PIL import Image
        row=(distance,margin,self.preprocess(Image.fromarray(square)),vector.copy())
        if close:
            self.closest[key]=row
            if len(self.closest)>8:del self.closest[max(self.closest,key=lambda k:self.closest[k][0])]
        if first:
            self.first[key]=row
            if len(self.first)>8:del self.first[max(self.first)]
    def verify(self,root,n_target):
        import numpy as np,torch
        import torch.nn.functional as functional
        rows=dict(self.first);rows.update(self.closest)
        if not rows:
            proof=dict(status="PASS_NO_HITS",samples=0,all_features_bitwise_equal=True,max_feature_error=0,route_decisions_equal=True)
        else:
            keys=sorted(rows)
            batch=torch.stack([rows[k][2] for k in keys]).to(device="cuda",dtype=self.model.visual.conv1.weight.dtype)
            with torch.inference_mode():actual=functional.normalize(self.model.encode_image(batch).float(),dim=-1).cpu().numpy()
            expected=np.stack([rows[k][3] for k in keys])
            errors=np.abs(actual-expected)
            old_scores=expected@self.text.T;new_scores=actual@self.text.T
            old_margin=np.max(old_scores[:,:n_target],axis=1)-np.max(old_scores[:,n_target:],axis=1)
            new_margin=np.max(new_scores[:,:n_target],axis=1)-np.max(new_scores[:,n_target:],axis=1)
            exact=bool(np.array_equal(actual,expected))
            proof=dict(status="PASS_EXACT" if exact else "FALLBACK_RECOMPUTE_ALL_A_FEATURE_HITS",
              samples=len(keys),sample_rule="8 closest to margin 0.03 plus 8 deterministic keys among actual B cache hits",
              all_features_bitwise_equal=exact,max_feature_error=float(errors.max()),max_margin_error=float(np.max(np.abs(old_margin-new_margin))),
              route_decisions_equal=bool(np.array_equal(old_margin>=self.threshold,new_margin>=self.threshold)),
              feature_keys=keys,cached_margins=old_margin.tolist(),recomputed_margins=new_margin.tolist(),
              source_feature_hits_considered=len(self.seen),semantic_decisions_reused=False)
        (root/"feature_numeric_cache_audit.json").write_text(json.dumps(proof,indent=2))
        return proof
