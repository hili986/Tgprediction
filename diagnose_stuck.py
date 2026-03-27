"""诊断卡住的分子 — 测试不同嵌入策略"""
import sys, time
sys.path.insert(0, '.')
from rdkit import Chem
from rdkit.Chem import AllChem, rdForceFieldHelpers
from src.features.virtual_polymerization import build_oligomer

smi = '*C#C[Si](C)(C)C#C[Si](C)(C)C#C[Si](C)(C)O[Si](C)(C)C#C[Si](C)(C)C#C[Si](*)(C)C'
olig = build_oligomer(smi, n=3)
mol = Chem.AddHs(Chem.MolFromSmiles(olig))
print('3-mer heavy atoms:', mol.GetNumHeavyAtoms())
print('UFF compatible:', rdForceFieldHelpers.UFFHasAllMoleculeParams(mol))

# === 策略1: srETKDGv3 关闭 torsion prefs ===
print('\n--- Strategy 1: srETKDGv3 basic knowledge ---')
t0 = time.time()
params = AllChem.srETKDGv3()
params.useRandomCoords = True
params.randomSeed = 42
params.useExpTorsionAnglePrefs = False
params.useBasicKnowledge = True
cids = AllChem.EmbedMultipleConfs(mol, numConfs=5, params=params)
t1 = time.time()
print(f'Result: {len(cids)} confs in {t1-t0:.1f}s')

# === 策略2: 纯随机坐标 DG ===
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

# === 策略3: ETKDGv3 + maxIterations 限制 ===
print('\n--- Strategy 3: ETKDGv3 with maxIterations=500 ---')
mol3 = Chem.AddHs(Chem.MolFromSmiles(olig))
t0 = time.time()
params3 = AllChem.ETKDGv3()
params3.useRandomCoords = True
params3.randomSeed = 42
params3.maxIterations = 500
cids3 = AllChem.EmbedMultipleConfs(mol3, numConfs=5, params=params3)
t1 = time.time()
print(f'Result: {len(cids3)} confs in {t1-t0:.1f}s')

print('\nDone!')
