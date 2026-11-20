"""Read-only audit of original EFA/Zooniverse files; never uses SEEK as identity truth."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from datetime import datetime
import json, os, warnings
import pandas as pd
from PIL import Image

ROOT=Path('/archive/vision/beery/animal_reid/datasets/elephants_zooniverse')
OUT=Path(__file__).resolve().parent

def date(s):
    try: return datetime.strptime(str(s).strip('\x00 '),'%Y:%m:%d %H:%M:%S').isoformat(sep=' ')
    except (ValueError,TypeError): return ''

def inspect(p):
    rel=p.relative_to(ROOT/'images'); parts=rel.parts
    r=dict(image=str(rel),collection=parts[0],folder_id=parts[1] if len(parts)>2 else '',bytes=p.stat().st_size)
    try:
        with warnings.catch_warnings(record=True) as ws:
            with Image.open(p) as im:
                r.update(format=im.format,width=im.width,height=im.height)
                ex=im._getexif() or {} if hasattr(im,'_getexif') else {}
                for name,tag in [('original',36867),('digitized',36868),('modified',306)]:
                    r[name+'_raw']=str(ex.get(tag,'')); r[name+'_date']=date(ex.get(tag,''))
                r['date_source']=next((n for n in ('original','digitized','modified') if r[n+'_date']),'')
                r['capture_date']=r.get(r['date_source']+'_date','')
                for name,tag in [('make',271),('model',272),('description',270),('software',305),('offset_original',36881)]: r[name]=str(ex.get(tag,''))
                gps=ex.get(34853,{})
                r['gps_tags']=str(gps)
                if isinstance(gps,dict) and all(t in gps for t in [1,2,3,4]):
                    def coord(v,ref): return (float(v[0])+float(v[1])/60+float(v[2])/3600)*(-1 if ref in ['S','W',b'S',b'W'] else 1)
                    r['latitude']=coord(gps[2],gps[1]);r['longitude']=coord(gps[4],gps[3])
            r['warnings']='; '.join(str(w.message) for w in ws)
    except Exception as e: r['error']=str(e)
    return r

if __name__=='__main__':
    manifest=[]
    for p in sorted(ROOT.glob('*manifest*.json')):
        for k,e in json.load(open(p)).items():
            for im in e['images']:
                manifest.append(dict(image=im,manifest_key=k,subject_id=e['info'].get('subject_id',''),upload_time=e['info'].get('upload_time',''),**{k.lstrip('#'):v for k,v in e['upload_metadata'].items() if not isinstance(v,list)}))
    mf=pd.DataFrame(manifest);mf.to_csv(OUT/'manifest.csv',index=False)
    paths=[]
    for parent,dirs,files in os.walk(ROOT/'images'):
        dirs[:]=sorted(d for d in dirs if not d.startswith('.'))
        paths.extend(Path(parent)/f for f in sorted(files) if not f.startswith('.') and not f.lower().endswith('.db'))
    print('Scanning',len(paths),'files',flush=True)
    records=[]
    with ThreadPoolExecutor(max_workers=12) as pool:
        for i,r in enumerate(pool.map(inspect,paths),1):
            records.append(r)
            if i%2000==0: print(i,'files inspected',flush=True)
    df=pd.DataFrame(records);df.to_csv(OUT/'image_inventory.csv',index=False)
    core=mf.merge(df,on='image',how='left',validate='one_to_one',indicator=True)
    core.to_csv(OUT/'core_images.csv',index=False)
    print(df.groupby('collection').size().to_string(),flush=True)
    print('Manifest file match:',core['_merge'].value_counts().to_dict(),flush=True)
