#!/usr/bin/env python3
"""T_world_base 업데이트 유틸리티.

Usage:
  python scripts/update_world_base.py \\
      --calib configs/calibration/extrinsic_20260612_170053.json \\
      --x 0.066 --y -0.122 --z 0.099 \\
      --roll 0.0 --pitch 0.0 --yaw 0.0

  --roll/pitch/yaw 단위: radians (기본값 0)
  --roll_deg/pitch_deg/yaw_deg 옵션으로 degrees 입력도 가능
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


def xyzrpy_to_T(x, y, z, roll, pitch, yaw) -> np.ndarray:
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    R  = Rz @ Ry @ Rx
    T  = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = [x, y, z]
    return T


def main():
    ap = argparse.ArgumentParser(description='Update T_world_base in calibration JSON')
    ap.add_argument('--calib', required=True, help='calibration JSON path')
    ap.add_argument('--x',     type=float, required=True)
    ap.add_argument('--y',     type=float, required=True)
    ap.add_argument('--z',     type=float, required=True)
    ap.add_argument('--roll',      type=float, default=None)
    ap.add_argument('--pitch',     type=float, default=None)
    ap.add_argument('--yaw',       type=float, default=None)
    ap.add_argument('--roll_deg',  type=float, default=None)
    ap.add_argument('--pitch_deg', type=float, default=None)
    ap.add_argument('--yaw_deg',   type=float, default=None)
    ap.add_argument('--dry_run', action='store_true', help='print only, do not save')
    args = ap.parse_args()

    roll  = args.roll  if args.roll  is not None else math.radians(args.roll_deg  or 0.0)
    pitch = args.pitch if args.pitch is not None else math.radians(args.pitch_deg or 0.0)
    yaw   = args.yaw   if args.yaw   is not None else math.radians(args.yaw_deg   or 0.0)

    calib_path = Path(args.calib)
    with open(calib_path) as f:
        data = json.load(f)

    T = xyzrpy_to_T(args.x, args.y, args.z, roll, pitch, yaw)
    print(f'새 T_world_base (roll={math.degrees(roll):.2f}°, pitch={math.degrees(pitch):.2f}°, yaw={math.degrees(yaw):.2f}°):')
    print(np.round(T, 9))

    old_params = data.get('_world_base_params', {})
    print(f'\n기존 params: {old_params}')
    new_params = dict(x=args.x, y=args.y, z=args.z,
                      roll_rad=round(roll, 6),
                      pitch_rad=round(pitch, 6),
                      yaw_rad=round(yaw, 6),
                      note=f'roll={math.degrees(roll):.2f}°, pitch={math.degrees(pitch):.2f}°, yaw={math.degrees(yaw):.2f}°. Updated by update_world_base.py')
    print(f'새 params:   {new_params}')

    if args.dry_run:
        print('\n[dry_run] 저장하지 않음.')
        return

    data['_world_base_params'] = new_params
    data['T_world_base']       = T.tolist()

    with open(calib_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f'\n[OK] {calib_path} 업데이트 완료.')


if __name__ == '__main__':
    main()
