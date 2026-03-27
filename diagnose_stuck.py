"""诊断卡住的分子 — 只测策略2和3 (策略1已确认卡死)"""
import sys, time
sys.path.insert(0, '.')
from rdkit import Chem
from rdkit.Chem import AllChem
from src.features.virtual_polymerization import build_oligomer

smi = '*C#C[Si](C)(C)C#C[Si](C)(C)C#C[Si](C)(C)O[Si](C)(C)C#C[Si](C)(C)C#C[Si](*)(C)C'
olig = build_oligomer(smi, n=3)
mol = Chem.AddHs(Chem.MolFromSmiles(olig))
print('3-mer heavy atoms:', mol.GetNumHeavyAtoms())

# === 策略2: 纯随机坐标 DG (完全绕过 ETKDG) ===
print('\n--- Strategy 2: Basic DG random coords ---')
mol2 = Chem.AddHs(Chem.MolFromSmiles(olig))
t0 = time.time()
params2 = AllChem.EmbedParameters()
params2.useRandomCoords = True
params2.randomSeed = 42
params2.maxIterations = 500
cids2 = AllChem.EmbedMultipleConfs(mol2, numConfs=5, params=params2)
t1 = time.time()
print(f'Result: {len(cids2)} confs in {t1-t0:.1f}s')
if len(cids2) > 0:
    # 验证 MMFF 能否优化
    props = AllChem.MMFFGetMoleculeProperties(mol2)
    print(f'MMFF props: {"OK" if props else "FAILED"}')

# === 策略3: ETKDGv3 + maxIterations=500 ===
print('\n--- Strategy 3: ETKDGv3 maxIterations=500 ---')
mol3 = Chem.AddHs(Chem.MolFromSmiles(olig))
t0 = time.time()
params3 = AllChem.ETKDGv3()
params3.useRandomCoords = True
params3.randomSeed = 42
params3.maxIterations = 500
cids3 = AllChem.EmbedMultipleConfs(mol3, numConfs=5, params=params3)
t1 = time.time()
print(f'Result: {len(cids3)} confs in {t1-t0:.1f}s')

# === 策略4: ETKDGv3 关闭所有高级选项 ===
print('\n--- Strategy 4: ETKDGv3 bare minimum ---')
mol4 = Chem.AddHs(Chem.MolFromSmiles(olig))
t0 = time.time()
params4 = AllChem.ETKDGv3()
params4.useRandomCoords = True
params4.randomSeed = 42
params4.useExpTorsionAnglePrefs = False
params4.useBasicKnowledge = False
params4.enforceChirality = False
params4.maxIterations = 200
cids4 = AllChem.EmbedMultipleConfs(mol4, numConfs=5, params=params4)
t1 = time.time()
print(f'Result: {len(cids4)} confs in {t1-t0:.1f}s')

print('\nDone!')
