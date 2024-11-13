## One-Hot Encoding Scheme for SEEK Code Attributes

**Gender**: 3 values
- `[1, 0, 0]` for `B`
- `[0, 1, 0]` for `C`
- `[0, 0, 1]` for `_`

**Age**: 2 values
- `[1, 0]` for `00`
- `[0, 1]` for `20`

**Tusk (L and R)**: 3 values
- `[1, 0, 0]` for `0`
- `[0, 1, 0]` for `_`
- `[0, 0, 1]` for `1`

**Left Ear (Tear/Hole)**: 5 values
- `[1, 0, 0, 0, 0]` for `0`
- `[0, 1, 0, 0, 0]` for `_`
- `[0, 0, 1, 0, 0]` for `3`
- `[0, 0, 0, 1, 0]` for `4`
- `[0, 0, 0, 0, 1]` for `5`

**Right Ear (Tear/Hole)**: 5 values
- `[1, 0, 0, 0, 0]` for `0`
- `[0, 1, 0, 0, 0]` for `_`
- `[0, 0, 1, 0, 0]` for `7`
- `[0, 0, 0, 1, 0]` for `8`
- `[0, 0, 0, 0, 1]` for `9`

**Extreme (L and R)**: 3 values
- `[1, 0, 0]` for `0`
- `[0, 1, 0]` for `_`
- `[0, 0, 1]` for `1`

**Special (Ear and Body)**: 3 values
- `[1, 0, 0]` for `0`
- `[0, 1, 0]` for `_`
- `[0, 0, 1]` for `1`
