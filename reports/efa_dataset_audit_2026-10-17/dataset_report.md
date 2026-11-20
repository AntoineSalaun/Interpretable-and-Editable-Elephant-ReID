# Elephant4Africa / Zooniverse dataset inventory

Audited 17 October 2026. This describes the local collection associated with the Zooniverse labels, excluding the Mara dataset. Original images and labels were only read; no SEEK codes were changed.

## What collection is this?

The source is `/archive/vision/beery/animal_reid/datasets/elephants_zooniverse/`. Its two complete upload manifests identify the core collection unambiguously:

- `EFA_IDI__complete__manifest_uploaded.json`: identified elephants.
- `EFA_IDU__complete__manifest_uploaded.json`: unidentified elephants.

The copies of these manifests inside `ele_id_project/data/` are identical to the top-level copies. Every one of the **8,692 manifest image paths exists locally**, with a readable image header. There are no duplicate paths or byte-identical image files within this core collection. This is not a full pixel-integrity or near-duplicate check.

| Measure | Identified (`EFA_IDI`) | Unidentified (`EFA_IDU`) | Combined |
|---|---:|---:|---:|
| Original image files | 5,504 | 3,188 | **8,692** |
| Existing elephant IDs / temporary groups | **1,008 elephant IDs** | **675 temporary groups** | 1,683 labels; **not 1,683 confirmed individuals** |
| Images with valid EXIF capture dates | 5,445 | 3,188 | 8,633 (99.3%) |
| Images without valid capture dates | 59 | 0 | 59 |
| Images with GPS coordinates | 10 | 0 | 10 |
| Current experimental table images | 5,348 | 3,096 | 8,444 |
| IDs / groups in experimental table | 999 | 671 | 1,670 labels |

Identified images account for 63.3% of the core collection; unidentified images account for 36.7%. The identified labels comprise 1,007 `B` IDs and one `F` ID. These are existing catalogue labels; this audit does not independently verify the identity assignments or infer sex from the prefixes.

The 675 `T` labels organize unidentified images. Their relationship to one another and to the identified catalogue is unverified. The accompanying project documentation explicitly says unidentified images may depict already identified elephants or new elephants. **The confirmed catalogue size available here is 1,008 IDs; the number of additional distinct elephants in the unidentified images cannot be established from these files.**

## Dates: original capture time, not file or edit time

All 8,633 valid capture dates come directly from EXIF `DateTimeOriginal`. No fallback dates were required. The timestamps have no recorded timezone offset, so they are reported as the camera recorded them, without conversion.

| Collection | Earliest capture | Latest capture |
|---|---|---|
| Identified | 4 May 2012, 08:44:19 | 16 June 2015, 11:53:49 |
| Unidentified | 26 September 2014, 12:34:05 | 15 August 2015, 17:04:47 |
| Combined | **4 May 2012** | **15 August 2015** |

| Capture year | Identified images | Unidentified images | Total |
|---|---:|---:|---:|
| 2012 | 111 | 0 | 111 |
| 2013 | 3,578 | 0 | 3,578 |
| 2014 | 1,686 | 1,467 | 3,153 |
| 2015 | 70 | 1,721 | 1,791 |
| Unknown / invalid | 59 | 0 | 59 |

The files cover 363 distinct calendar dates overall: 290 for identified images and 85 for unidentified images, with some overlap. This is a collection spanning these dates, not evidence of continuous sampling. Most identified images are from 2013–2014, whereas unidentified images are from late 2014–2015.

Of the 59 undated images, 26 have no original-date value and 33 contain `0000:00:00 00:00:00`. They belong to 29 identified elephants; eight IDs have no usable dated image at all.

**The old preprocessing notebook extracted EXIF `DateTime`, which represents modification time.** It differs from `DateTimeOriginal` for 583 core images: 339 identified and 244 unidentified. Within the 8,444-image experimental table, 562 stored `picture_time` values differ from capture time. This explains why that table appears to extend identified images to 22 August 2015 and starts unidentified images on 28 September 2014. For describing collection dates or reconstructing encounters, use the original capture timestamps above. Original camera timestamps can still contain clock errors; no field log was available to validate them.

## How many sightings?

**There is no authoritative sighting identifier in these manifests.** `#site` holds the elephant ID or temporary group; it is not a geographical site. Every `#roll` equals `1`. The source documentation states that the `B1000_R1`-style folders were added to fit the upload pipeline, so these folders do not establish encounters. `#capture` indexes individual image records.

For an explicit working estimate, sort images separately within each existing ID / temporary group by EXIF capture time, and start a new encounter whenever the gap from the previous image exceeds 60 minutes. This produces:

| Derived measure | Identified | Unidentified | Total |
|---|---:|---:|---:|
| **Inferred individual encounters, 60-minute gap** | **1,844** | **687** | **2,531** |
| Distinct ID/group–calendar-day combinations | 1,781 | 678 | 2,459 |
| Median images per inferred encounter | 3 | 4 | — |

These estimates cover the 8,633 dated images only. The 59 undated images remain unassigned; they could belong to existing encounters or additional ones. These are individual-level photo encounters, not independent herd sightings or field visits. Simultaneously photographed elephants contribute separate individual encounters. Splitting or merging temporary `T` groups after identification could change the estimate.

Sensitivity to the chosen time gap:

| New encounter after a gap exceeding | Identified | Unidentified | Total |
|---|---:|---:|---:|
| 15 minutes | 2,146 | 714 | 2,860 |
| 30 minutes | 1,935 | 694 | 2,629 |
| **60 minutes** | **1,844** | **687** | **2,531** |
| 120 minutes | 1,799 | 684 | 2,483 |

Under the 60-minute rule, 349 identified elephants have at least two dated encounters. Their folders contain 3,420 images, including any undated images. Another 651 identified elephants have one inferred dated encounter, and eight have none. Among the unidentified groups, 12 have multiple inferred encounters. Kate's field encounter records would be preferable to these estimates.

## Images per individual and image characteristics

| Measure | Identified elephant IDs | Unidentified temporary groups |
|---|---:|---:|
| Mean images per ID/group | 5.46 | 4.72 |
| Median | 4 | 4 |
| Middle 50% | 3–7 | 3–6 |
| Range | 1–40 | 1–24 |
| IDs/groups with only one image | 71 | 19 |

The original core files occupy **22.74 GB** (decimal units). Identified images range from 455 to 5,184 pixels wide, and unidentified images from 486 to 6,000 pixels wide. Images have varying dimensions and aspect ratios; the separate 224×224 experimental derivatives are not the original dataset.

EXIF metadata records **16 different camera models** across the core collection; 26 images have no camera model recorded. Counts below refer to model names, not distinct physical camera units.

| Camera model (EXIF) | Identified images | Unidentified images | Total |
|---|---:|---:|---:|
| Canon EOS 650D | 375 | 2,075 | 2,450 |
| NIKON D5100 | 2,367 | 0 | 2,367 |
| Canon EOS 50D | 1,786 | 484 | 2,270 |
| SP570UZ | 818 | 8 | 826 |
| Canon EOS 400D DIGITAL | 6 | 347 | 353 |
| NIKON D3300 | 0 | 141 | 141 |
| NIKON D3200 | 0 | 89 | 89 |
| DMC-FZ38 | 40 | 0 | 40 |
| Canon EOS 7D | 38 | 0 | 38 |
| SLT-A58 | 0 | 28 | 28 |
| Canon EOS 60D | 28 | 0 | 28 |
| Not recorded | 26 | 0 | 26 |
| NIKON D5200 | 0 | 16 | 16 |
| HC600 HYPERFIRE | 11 | 0 | 11 |
| DMC-ZS1 | 4 | 0 | 4 |
| NIKON D300S | 4 | 0 | 4 |
| DMC-FS3 | 1 | 0 | 1 |

The identified collection is dominated by Nikon D5100 and Canon EOS 50D images, whereas the unidentified collection is dominated by Canon EOS 650D images. This difference in camera composition is relevant when evaluating deployment performance. Eleven identified images report a Reconyx HC600 HYPERFIRE camera. The source README describes camera-trap and focal-follow imagery, but there is no reliable per-image acquisition-method label for all images. Camera counts are also saved in `camera_models.csv`.

There are **220 valid image files with unconventional filename extensions** in `EFA_IDI`, such as `.23` or `._rightjpg`; a scan that only accepts `.jpg` would omit them. Pillow identifies the core collection as 8,671 JPEG, 20 MPO, and one PNG file. The PNG has a `.JPG` filename. Preserve the manifest paths or inspect file contents when importing.

## Locations

Only **10 images (0.12%)** contain actual latitude/longitude values. Many other images contain a GPS-version tag without coordinates; those do not supply a location.

The geotagged images cover three identified elephants, `B1009`, `B1070`, and `B1613`, on 1 and 4 February 2013. Their coordinates lie approximately between **20.39325°S and 20.39035°S**, and **24.49316°E and 24.51603°E**, in northern Botswana. Full values are retained in `gps_images.csv`.

Elephants for Africa describes its research area as the Boteti River / Makgadikgadi Pans National Park, Botswana ([official study-area page](https://www.elephantsforafrica.org/about-elephants-for-africa/study-sites/)). That provides project context, not proof of the origin of every photograph in this collection. Ask Kate to confirm the dataset's geographical coverage and supply a field-location table if available. The manifests do not provide meaningful geographical site names.

## Zooniverse provenance and the experimental subset

The manifests record uploads on **29 September 2023** for 3,188 unidentified images and **5 October 2023** for 5,503 identified images. One additional identified image is present locally and in the manifest but marked `uploaded: false`, with no Zooniverse subject ID: `EFA_IDI/B1002/B1002_R1/EFA_IDI_B1002_R1_IMAG0011.JPG`. Thus there are **8,691 uploaded subjects and 8,692 local core images**. Upload dates are not photograph dates or evidence of when Kate sent the data.

General and special workflow exports each cover all 8,691 uploaded core subjects; ear workflow annotations cover 8,573. The general workflow also contains three subjects outside these manifests. Consequently, counting annotation rows or subject IDs alone does not reproduce the number of local core images.

All three current `image_dictonary_{original,optimized,majority_vote}.csv` tables contain the same **8,444 image paths**. They omit **248 original images**: 156 identified and 92 unidentified. The omitted set consists of the 59 undated images plus 189 dated images whose image-level SEEK strings fail the preprocessing validity criterion. Two of the undated images also lack a valid SEEK string. Nine identified IDs and four unidentified groups disappear entirely from the experimental tables.

Because the proposed work will re-predict and correct SEEK codes, **use the full 8,692-image manifest collection as the starting inventory** and retain missing-date flags. The earlier label-based filtering need not define the deployment collection. Keep the existing identity labels separate from the regenerated SEEK attributes, with unidentified groups explicitly marked unresolved.

The historical `Project_Documentation.md` reports 5,503 identified images, 3,191 unidentified images, and 1,008 identified elephants. The inspected manifests and files establish the current local counts above; quote those to Kate. Some old summary statistics also disagree with the current files (for example, the measured median is four images per identified ID).

## Other material in the same archive

The parent archive also contains **25,204 JPG images under `images/EFA_S1ID/B1/`** and **985 non-hidden line-drawing images under `images/EleLineDrawings/`**. Hidden/checkpoint copies and `.db` files are excluded from these auxiliary counts. The `EFA_S1ID` files have no usable EXIF capture dates and no identity mapping in the two manifests audited here. The fact that they sit in a `B1` directory does not establish that they depict one individual.

These auxiliary files are excluded from all core totals above. Their relationship to the core collection, including possible visual overlap, has not been established. If Kate remembers a larger delivery, ask specifically about `EFA_S1ID` and whether a separate manifest exists.

## Files and reproducibility

Run `python reports/efa_dataset_audit_2026-10-17/audit.py` from the repository root, then `python reports/efa_dataset_audit_2026-10-17/summarize.py`.

The audit reads image headers and EXIF directly from source files, cross-checks both manifests, and checks potential byte duplicates by size and SHA-256. The analysis scripts require Python, Pillow, pandas, and matplotlib. The original data are never written.

Outputs in this folder include:

- `core_images_with_encounters.csv`: all 8,692 images, identity/group labels, EXIF dates, GPS, and inferred encounter assignment where available.
- `identity_summary.csv` and `inferred_encounters_60min.csv`: per-ID/group and per-encounter summaries.
- `images_by_year.csv`, `images_by_month.csv`, and `encounter_sensitivity.csv`.
- `missing_capture_dates.csv`, `excluded_from_experiments.csv`, and `capture_vs_modified_date_differences.csv`.
- `gps_images.csv`, `image_inventory.csv`, `manifest.csv`, `summary.json`, and `byte_identical_images.csv` (empty, with column headers).
- `dataset_overview.png` and `dataset_overview.pdf`.
- `email_draft.md`: suggested email to Kate, not sent.

Primary local references: the two original manifests; `ele_id_project/Project_Documentation.md` (Data Description); `ele_id_project/README.md` (Project Pipeline and Data Processing); and `data/Elephant4Africa/Image_preprocessing.ipynb` (filtering and date extraction).
