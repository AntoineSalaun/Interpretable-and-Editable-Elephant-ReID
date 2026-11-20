"""Produce reproducible summaries from audit.py outputs (no source data changes)."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import hashlib,json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
REPO=P.parents[1]
ROOT=Path('/archive/vision/beery/animal_reid/datasets/elephants_zooniverse')
d=pd.read_csv(P/'core_images.csv',low_memory=False)
a=pd.read_csv(P/'image_inventory.csv',low_memory=False)
camera_models=d['model'].fillna('').str.strip().replace('', 'Not recorded')
cameras=pd.crosstab(camera_models,d.collection).reindex(columns=['EFA_IDI','EFA_IDU'],fill_value=0)
cameras.index.name='camera_model'
cameras['Total']=cameras.sum(axis=1)
cameras.sort_values('Total',ascending=False).to_csv(P/'camera_models.csv')
d['dt']=pd.to_datetime(d.capture_date)
d['day']=d.dt.dt.strftime('%Y-%m-%d')
d=d.sort_values(['collection','site','dt','image'])
v=d.dropna(subset=['dt']).copy()
# An inferred encounter starts when consecutive images within one existing ID/group
# are more than 60 minutes apart. Never merges distinct identities using clock times.
v['encounter_number']=v.groupby(['collection','site']).dt.transform(lambda x:(x.diff().isna()|x.diff().gt(pd.Timedelta(hours=1))).cumsum())
v['inferred_encounter_id']=v.collection+'_'+v.site+'_'+v.encounter_number.astype(str)
sights=v.groupby(['collection','site','inferred_encounter_id']).agg(images=('image','size'),start=('dt','min'),end=('dt','max')).reset_index()
sights.to_csv(P/'inferred_encounters_60min.csv',index=False)
d=d.merge(v[['image','inferred_encounter_id']],on='image',how='left',validate='one_to_one')
d.to_csv(P/'core_images_with_encounters.csv',index=False)
ids=d.groupby(['collection','site']).agg(images=('image','size'),dated_images=('dt','count'),first_date=('dt','min'),last_date=('dt','max'),dated_days=('day','nunique'),inferred_encounters_60min=('inferred_encounter_id','nunique')).reset_index()
ids.to_csv(P/'identity_summary.csv',index=False)
years=pd.crosstab(d.dt.dt.year.fillna(-1).astype(int),d.collection).rename(index={-1:'Unknown'})
years.to_csv(P/'images_by_year.csv')
month=pd.crosstab(d.dt.dt.strftime('%Y-%m'),d.collection);month.to_csv(P/'images_by_month.csv')
old=pd.read_csv(REPO/'data/Elephant4Africa/out_apr2/image_dictonary_original.csv')
excluded=d[~d.image.isin(old.image)].copy();excluded.to_csv(P/'excluded_from_experiments.csv',index=False)
d[d.dt.isna()].to_csv(P/'missing_capture_dates.csv',index=False)
d[d.latitude.notna()].to_csv(P/'gps_images.csv',index=False)
d[d.original_date.notna()&d.modified_date.notna()&(d.original_date!=d.modified_date)].to_csv(P/'capture_vs_modified_date_differences.csv',index=False)
# Only same-byte-size files can be byte-identical, so hash candidates only.
candidates=d[d.bytes.duplicated(keep=False)].copy()
def digest(row):
 h=hashlib.sha256()
 with open(ROOT/'images'/row.image,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
with ThreadPoolExecutor(max_workers=8) as pool:candidates['sha256']=list(pool.map(digest,candidates.itertuples()))
dup=candidates[candidates.sha256.duplicated(keep=False)]
dup.to_csv(P/'byte_identical_images.csv',index=False)
summary={}
for name,g in d.groupby('collection'):
 gi=ids[ids.collection==name];gv=g.dropna(subset='dt'); gs=sights[sights.collection==name]
 summary[name]={'images':len(g),'ids_or_groups':g.site.nunique(),'dated_images':g.dt.notna().sum(),'undated_images':g.dt.isna().sum(),'first_capture':str(g.dt.min()),'last_capture':str(g.dt.max()),'calendar_days':g.day.nunique(),'identity_days':gi.dated_days.sum(),'inferred_encounters_60min':len(gs),'ids_with_2plus_encounters':gi.inferred_encounters_60min.ge(2).sum(),'images_in_ids_with_2plus_encounters':gi[gi.inferred_encounters_60min.ge(2)].images.sum(),'ids_with_no_date':gi.dated_images.eq(0).sum(),'single_image_ids':gi.images.eq(1).sum(),'images_per_id':gi.images.describe().to_dict(),'images_per_encounter':gs.images.describe().to_dict(),'gps_images':g.latitude.notna().sum(),'total_bytes':g.bytes.sum(),'excluded_from_experiments':len(excluded[excluded.collection==name]),'excluded_undated':len(excluded[(excluded.collection==name)&excluded.dt.isna()]),'non_jpg_extensions':(~g.image.str.lower().str.endswith('.jpg')).sum()}
summary['duplicate_check']={'candidate_files_hashed':len(candidates),'duplicate_files':len(dup),'duplicate_groups':dup.sha256.nunique(),'redundant_files':len(dup)-dup.sha256.nunique()}
summary['overall']={'distinct_capture_days':d.day.nunique(),'total_bytes':d.bytes.sum(),'undated':d.dt.isna().sum(),'original_modified_mismatches':len(pd.read_csv(P/'capture_vs_modified_date_differences.csv'))}
sens=[]
for gap in [15,30,60,120]:
 row={'gap_minutes':gap}
 for name,g in v.groupby('collection'):
  row[name]=int((g.groupby('site').dt.diff().isna()|g.groupby('site').dt.diff().gt(pd.Timedelta(minutes=gap))).sum())
 row['total']=row['EFA_IDI']+row['EFA_IDU'];sens.append(row)
pd.DataFrame(sens).to_csv(P/'encounter_sensitivity.csv',index=False)
(P/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.item()))
print(json.dumps(summary,indent=2,default=lambda x:x.item()))
print('Sensitivity',sens)
print('Excluded dated',excluded[excluded.dt.notna()].groupby('collection').size().to_dict())
print('Old date != original',int((pd.to_datetime(old.picture_time).to_numpy()!=pd.to_datetime(old.image.map(d.set_index('image').capture_date)).to_numpy()).sum()))
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
years.reindex([2012,2013,2014,2015,'Unknown']).rename(columns={'EFA_IDI':'Identified','EFA_IDU':'Unidentified'}).plot.bar(ax=axes[0],color=['#246a96','#df8a36'],rot=0)
axes[0].set(xlabel='EXIF capture year',ylabel='Images',title='8,692 images in the Zooniverse collection')
for name,color,label in [('EFA_IDI','#246a96','Identified IDs'),('EFA_IDU','#df8a36','Unidentified groups')]:
 s=ids[ids.collection==name].images.value_counts().sort_index();axes[1].plot(s.index,s.values,marker='o',markersize=3,color=color,label=label)
axes[1].set(xlabel='Images per ID / temporary group',ylabel='IDs / groups',title='1,008 identified IDs; 675 temporary groups',yscale='log')
axes[1].legend();fig.tight_layout();fig.savefig(P/'dataset_overview.png',dpi=180);fig.savefig(P/'dataset_overview.pdf');plt.close(fig)
