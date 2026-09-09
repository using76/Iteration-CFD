#!/usr/bin/env python3
# meteor-cfd - Copyright (c) 2026 주식회사 이터레이션즈 (Iterations Co., Ltd.)
# Source-available, not Open Source. See LICENSE at the repository root.
# No GPL-licensed source was consulted.
# Write a copy of a Fluent ASCII mesh with every face's node order reversed (c0/c1 untouched):
# the orientation convention foamMeshToFluent uses (right-hand normal toward c0, boundary
# normals pointing into the cell) instead of the manual's (normal toward c1).
#   python fluent_invert.py <in.msh> <out.msh>
import re, sys, time
src, dst = sys.argv[1], sys.argv[2]
t0 = time.time()
hdr = re.compile(r'^\(13\s*\(\s*([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s+([0-9a-fA-F]+)\s*\)\s*\(?\s*$')
n_faces = 0; n_blocks = 0
with open(src, 'r', encoding='ascii', errors='replace', newline='') as f, open(dst, 'w', encoding='ascii', newline='') as g:
    in_faces = False; etype = 0
    for line in f:
        if not in_faces:
            m = hdr.match(line.rstrip('\r\n'))
            if m and m.group(1) != '0':                 # a real faces block (zone id != 0)
                in_faces = True; etype = int(m.group(5), 16); n_blocks += 1
            g.write(line)
            continue
        s = line.strip()
        if s.startswith(')'):
            in_faces = False
            g.write(line)
            continue
        parts = s.split()
        if not parts:
            g.write(line); continue
        if etype == 0:                                  # mixed: count first
            k = int(parts[0], 16)
            nodes = parts[1:1 + k]; rest = parts[1 + k:]
            g.write(' '.join([parts[0]] + nodes[::-1] + rest) + '\n')
        else:
            k = etype                                   # 2 = line, 3 = tri, 4 = quad
            nodes = parts[:k]; rest = parts[k:]
            g.write(' '.join(nodes[::-1] + rest) + '\n')
        n_faces += 1
print('faces reversed: %d in %d blocks, %.0f s -> %s' % (n_faces, n_blocks, time.time() - t0, dst))
