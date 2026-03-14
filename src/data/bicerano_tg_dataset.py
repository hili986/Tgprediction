"""
Bicerano Tg Dataset — 304 Linear Homopolymers with Glass Transition Temperature
Bicerano 玻璃化转变温度数据集 — 304 种线性均聚物

Source / 数据来源:
    Choi et al., Scientific Data 11, 363 (2024)
    DOI: 10.1038/s41597-024-03212-4
    Figshare: 10.6084/m9.figshare.c.6858337.v1
    Original: Bicerano, "Prediction of Polymer Properties", 3rd ed., 2002

Each entry: (polymer_name, repeat_unit_smiles, bigsmiles, tg_kelvin)
BigSMILES strings use bracket notation compatible with bigsmiles_checker.py.
SMILES use * for attachment points (repeat unit fragment).

Public API / 公共 API:
    BICERANO_DATA        — raw tuple list (304 entries)
    load_dataset()       — list of dicts
    get_names()          — polymer names
    get_smiles()         — repeat unit SMILES
    get_bigsmiles()      — BigSMILES strings
    get_tg_values()      — Tg values in Kelvin
    to_csv(path)         — export to CSV
    to_json(path)        — export to JSON
    validate_all()       — validate all BigSMILES with checker
    shorthand_to_bracket — convert shorthand BigSMILES to bracket notation
"""

import csv
import json
from typing import List, Dict, Any, Tuple


# ---------------------------------------------------------------------------
# Raw data: (name, repeat_unit_smiles, bigsmiles, tg_kelvin)
# ---------------------------------------------------------------------------

BICERANO_DATA: Tuple[Tuple[str, str, str, int], ...] = (
    ('Poly[oxy(diethylsilylene)]', '*O[Si](*)(CC)CC', '{[<]O[Si](CC)(CC)[>]}', 130),
    ('Poly(dimethyl siloxane)', '*O[Si](*)(C)C', '{[<]O[Si](C)(C)[>]}', 152),
    ('Poly(cis-1,4-butadiene)', '*C/C=C\\C*', '{[$]C/C=C\\C[$]}', 171),
    ('Poly(dimethylsilylenemethylene)', '*C[Si](*)(C)C', '{[<]C[Si](C)(C)[>]}', 173),
    ('Poly[oxy(methylphenylsilylene)]', '*O[Si](*)(C)c1ccccc1', '{[<]O[Si](c1ccccc1)(C)[>]}', 187),
    ('Poly(3-hexoxypropylene oxide)', '*CC(COCCCCCC)O*', '{[<]CC(COCCCCCC)O[>],[<]C(COCCCCCC)CO[>]}', 188),
    ('Polyoxytetramethylene', '*CCCCO*', '{[<]CCCCO[>]}', 190),
    ('Poly(1,1-dimethylsilazane)', '*N[Si](*)(C)C', '{[<]N[Si](C)(C)[>]}', 191),
    ('Poly(2-n-butyl-1,4-butadiene)', '*CC=C(C*)CCCC', '{[$]CC=C(CCCC)C[$]}', 192),
    ('Poly(vinyl n-octyl ether)', '*CC(*)OCCCCCCCC', '{[$]CC(OCCCCCCCC)[$]}', 194),
    ('Poly(3-butoxypropylene oxide)', '*CC(COCCCC)O*', '{[<]CC(COCCCC)O[>],[<]C(COCCCC)CO[>]}', 194),
    ('Polyethylene', '*CC*', '{[$]CC[$]}', 195),
    ('Polyoxytrimethylene', '*CCCO*', '{[<]CCCO[>]}', 195),
    ('Poly(2-n-propyl-1,4-butadiene)', '*CC=C(C*)CCC', '{[$]CC=C(CCC)C[$]}', 196),
    ('Poly(vinyl n-decyl ether)', '*CC(*)OCCCCCCCCCC', '{[$]CC(OCCCCCCCCCC)[$]}', 197),
    ('Poly(2-ethyl-1,4-butadiene)', '*CC=C(C*)CC', '{[$]CC=C(CC)C[$]}', 197),
    ('Poly[oxy(methyl ??-trifluoropropylsilylene)]', '*O[Si](*)(C)CCC(F)(F)F', '{[<]O[Si](CCC(F)(F)F)(C)[>]}', 199),
    ('Polyisobutylene', '*CC(*)(C)C', '{[$]CC(C)(C)[$]}', 199),
    ('Poly(dimethylsilylenetrimethylene)', '*CCC[Si](*)(C)C', '{[<]CCC[Si](C)(C)[>]}', 203),
    ('Polyoxyoctamethylene', '*CCCCCCCCO*', '{[<]CCCCCCCCO[>]}', 203),
    ('Polyoxyhexamethylene', '*CCCCCCO*', '{[<]CCCCCCO[>]}', 204),
    ('Poly(tetramethylene adipate)', '*OCCCCOC(=O)CCCCC(*)=O', '{[<]C(=O)CCCCC(=O)[<],[>]OCCCCO[>]}', 205),
    ('Polyoxyethylene', '*CCO*', '{[<]CCO[>]}', 206),
    ('Poly(propylene oxide)', '*CC(C)O*', '{[<]CC(C)O[>],[<]C(C)CO[>]}', 206),
    ('Poly(vinyl n-pentyl ether)', '*CC(*)OCCCCC', '{[$]CC(OCCCCC)[$]}', 207),
    ('Poly(vinyl 2-ethylhexyl ether)', '*CC(*)OCC(CC)CCCC', '{[$]CC(OCC(CC)CCCC)[$]}', 207),
    ('Poly(n-octyl acrylate)', '*CC(*)C(=O)OCCCCCCCC', '{[$]CC(C(=O)OCCCCCCCC)[$]}', 208),
    ('Poly(vinyl n-hexyl ether)', '*CC(*)OCCCCCC', '{[$]CC(OCCCCCC)[$]}', 209),
    ('Poly(3-methoxypropylene oxide)', '*CC(COC)O*', '{[<]CC(COC)O[>],[<]C(COC)CO[>]}', 211),
    ('Poly(n-heptyl acrylate)', '*CC(*)C(=O)OCCCCCCC', '{[$]CC(C(=O)OCCCCCCC)[$]}', 213),
    ('Poly(??-caprolactone)', '*OCCCCCC(*)=O', '{[<]OCCCCCC(=O)[>]}', 213),
    ('Poly(n-nonyl acrylate)', '*CC(*)C(=O)OCCCCCCCCC', '{[$]CC(C(=O)OCCCCCCCCC)[$]}', 215),
    ('Poly(n-hexyl acrylate)', '*CC(*)C(=O)OCCCCCC', '{[$]CC(C(=O)OCCCCCC)[$]}', 216),
    ('Poly(decamethylene adipate)', '*OCCCCCCCCCCOC(=O)CCCCC(*)=O', '{[<]C(=O)CCCCC(=O)[<],[>]OCCCCCCCCCCO[>]}', 217),
    ('Polyoxymethylene', '*CO*', '{[<]CO[>]}', 218),
    ('Poly(n-dodecyl methacrylate)', '*CC(*)(C)C(=O)OCCCCCCCCCCCC', '{[$]CC(C(=O)OCCCCCCCCCCCC)(C)[$]}', 218),
    ('Poly(n-butyl acrylate)', '*CC(*)C(=O)OCCCC', '{[$]CC(C(=O)OCCCC)[$]}', 219),
    ('Poly(1-heptene)', '*CC(*)CCCC', '{[$]CC(CCCCC)[$]}', 220),
    ('Poly(oxycarbonyl-3- methylpentamethylene)', '*OCCC(C)CCC(*)=O', '{[<]OCCC(C)CCC(=O)[>]}', 220),
    ('Poly(vinyl n-butyl ether)', '*CC(*)OCCCC', '{[$]CC(OCCCC)[$]}', 221),
    ('Poly(2-isopropyl-1,4-butadiene)', '*CC=C(C*)C(C)C', '{[$]CC=C(C(C)C)C[$]}', 221),
    ('Poly(l-hexene)', '*CC(*)CCCC', '{[$]CC(CCCC)[$]}', 223),
    ('Poly(l-pentene)', '*CC(*)CCC', '{[$]CC(CCC)[$]}', 223),
    ('Polychloroprene', '*CC=C(Cl)C*', '{[$]CC=C(Cl)C[$]}', 225),
    ('Poly(propylene sulfide)', '*CC(C)S*', '{[<]CC(C)S[>]}', 226),
    ('Poly(1-butene)', '*CC(*)CC', '{[$]CC(CC)[$]}', 228),
    ('Poly(ethylene azelate)', '*OCCOC(=O)CCCCCCCC(*)=O', '{[<]C(=O)CCCCCCCC(=O)[<],[>]OCCO[>]}', 228),
    ('Poly(2-octyl acrylate)', '*CC(*)C(=O)OC(C)CCCCCC', '{[$]CC(C(=O)OC(C)CCCCCC)[$]}', 228),
    ('Poly(n-propyl acrylate)', '*CC(*)C(=O)OCCC', '{[$]CC(C(=O)OCCC)[$]}', 229),
    ('Polypropylene', '*CC(*)C', '{[$]CC(C)[$]}', 233),
    ('Poly(vinylidene fluoride)', '*CC(*)(F)F', '{[$]CC(F)(F)[$]}', 233),
    ('Poly(ethylene adipate)', '*OCCOC(=O)CCCCC(*)=O', '{[<]C(=O)CCCCC(=O)[<],[>]OCCO[>]}', 233),
    ('Poly(2-heptyl acrylate)', '*CC(*)C(=O)OC(C)CCCCC', '{[$]CC(C(=O)OC(C)CCCCC)[$]}', 235),
    ('Poly(6-methyl-1-heptene)', '*CC(*)CCCC(C)C', '{[$]CC(CCCC(C)C)[$]}', 239),
    ('Poly(oxycarbonyl-1,5- dimethylpentamethylene)', '*OC(C)CCCC(C)C(*)=O', '{[<]OC(C)CCCC(C)C(=O)[>]}', 240),
    ('Poly(2-bromo-1,4-butadiene)', '*CC=C(Br)C*', '{[$]CC=C(Br)C[$]}', 241),
    ('Poly(ethylene sebacate)', '*OCCOC(=O)CCCCCCCCC(*)=O', '{[<]C(=O)CCCCCCCCC(=O)[<],[>]OCCO[>]}', 243),
    ('Poly[(methyl)phenylsilylenetrimethylene]', '*CCC[Si](*)(C)c1ccccc1', '{[<]CCC[Si](c1ccccc1)(C)[>]}', 243),
    ('Poly(isobutyl acrylate)', '*CC(*)C(=O)OCC(C)C', '{[$]CC(C(=O)OCC(C)C)[$]}', 249),
    ('Poly(vinyl isobutyl ether)', '*CC(*)OCC(C)C', '{[$]CC(OCC(C)C)[$]}', 251),
    ('Poly(ethyl acrylate)', '*CC(*)C(=O)OCC', '{[$]CC(C(=O)OCC)[$]}', 251),
    ('Poly(n-octyl methacrylate)', '*CC(*)(C)C(=O)OCCCCCCCC', '{[$]CC(C(=O)OCCCCCCCC)(C)[$]}', 253),
    ('Poly(vinyl sec-butyl ether)', '*CC(*)OC(C)CC', '{[$]CC(OC(C)CC)[$]}', 253),
    ('Poly(sec-butyl acrylate)', '*CC(*)C(=O)OC(C)CC', '{[$]CC(C(=O)OC(C)CC)[$]}', 253),
    ('Poly(vinyl ethyl ether)', '*CC(*)OCC', '{[$]CC(OCC)[$]}', 254),
    ('Perfluoropolymer', '*c1nc(C(F)(F)F)nc(C(F)(OC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)OC(*)(F)C(F)(F)F)C(F)(F)F)n1', '{[<]c1nc(C)nc(n1)C(F)(C(F)(F)F)OC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)OC(C(F)(F)F)[>]}', 255),
    ('Poly(vinylidene chloride)', '*CC(*)(Cl)Cl', '{[$]CC(Cl)(Cl)[$]}', 256),
    ('Poly(3-pentyl acrylate)', '*CC(*)C(=O)OC(CC)CC', '{[$]CC(C(=O)OC(CC)CC)[$]}', 257),
    ('Poly(5-methyl-1-hexene)', '*CC(*)CCC(C)C', '{[$]CC(CCC(C)C)[$]}', 259),
    ('Perfluoropolymer', '*c1nc(C(F)(F)C(F)(F)OC(F)(F)F)nc(C(F)(F)C(F)(F)C(F)(F)C(*)(F)F)n1', '{[<]c1nc(nc(C(F)(F)F)n1)C(C(F)(F)F)OC(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)OC(C(F)(F)F)[>]}', 260),
    ('Poly(oxy-2,2- dichloromethyltrimethylene)', '*CC(CCl)(CCl)CO*', '{[<]CC(CCl)(CCl)CO[>]}', 265),
    ('Poly[(4-dimethylaminophenyl) methylsilylenetrimethylene]', '*CCC[Si](*)(C)c1ccc(N(C)C)cc1', '{[<]CCC[Si](c1ccc(N(C)C)cc1)(C)[>]}', 267),
    ('Poly(n-hexyl methacrylate)', '*CC(*)(C)C(=O)OCCCCCC', '{[$]CC(C(=O)OCCCCCC)(C)[$]}', 268),
    ('Poly(1,2-butadiene)', '*CC(*)C=C', '{[$]CC(C=C)[$]}', 269),
    ('Poly(vinyl isopropyl ether)', '*CC(*)OC(C)C', '{[$]CC(OC(C)C)[$]}', 270),
    ('Poly(ethylene succinate)', '*OCCOC(=O)CCC(*)=O', '{[<]C(=O)CCC(=O)[<],[>]OCCO[>]}', 272),
    ('Poly(vinyl methyl sulfide)', '*CC(*)SC', '{[$]CC(SC)[$]}', 272),
    ('Poly(oxydiphenylsilylene oxydimethylsilylene-1,4- phenylenedimethylsilylene)', '*O[Si](O[Si](C)(C)c1ccc([Si](*)(C)C)cc1)(c1ccccc1)c1ccccc1', '{[<]O[Si](c1ccccc1)(c1ccccc1)O[Si](C)(C)c1ccc(cc1)[Si](C)(C)[>]}', 273),
    ('Poly(vinyl butyrate)', '*CC(*)OC(=O)CCC', '{[$]CC(OC(=O)CCC)[$]}', 278),
    ('Poly(p-n-hexoxymethyl styrene)', '*CC(*)c1ccc(COCCCCCC)cc1', '{[$]CC(c1ccc(COCCCCCC)cc1)[$]}', 278),
    ('Poly(p-n-butyl styrene)', '*CC(*)c1ccc(CCCC)cc1', '{[$]CC(c1ccc(CCCC)cc1)[$]}', 279),
    ('Poly(methyl acrylate)', '*CC(*)C(=O)OC', '{[$]CC(C(=O)OC)[$]}', 281),
    ('Poly(vinyl propionate)', '*CC(*)OC(=O)CC', '{[$]CC(OC(=O)CC)[$]}', 283),
    ('Poly(2-ethylbutyl methacrylate)', '*CC(*)(C)C(=O)OCC(CC)CC', '{[$]CC(C(=O)OCC(CC)CC)(C)[$]}', 284),
    ('Poly(o-n-octoxy styrene)', '*CC(*)c1ccccc1OCCCCCCCC', '{[$]CC(c1ccccc1OCCCCCCCC)[$]}', 286),
    ('Poly(2-t-butyl-1,4-butadiene)', '*CC=C(C*)C(C)(C)C', '{[$]CC=C(C(C)(C)C)C[$]}', 293),
    ('Poly(n-butyl methacrylate)', '*CC(*)(C)C(=O)OCCCC', '{[$]CC(C(=O)OCCCC)(C)[$]}', 293),
    ('Poly(2-methoxyethyl methacrylate)', '*CC(*)(C)C(=O)OCCOC', '{[$]CC(C(=O)OCCOC)(C)[$]}', 293),
    ('Poly(p-n-propoxymethyl styrene)', '*CC(*)c1ccc(COCCC)cc1', '{[$]CC(c1ccc(COCCC)cc1)[$]}', 295),
    ('Poly(ethyl-p-xylylene)', '*Cc1ccc(C*)c(CC)c1', '{[$]Cc1ccc(c(CC)c1)C[$]}', 298),
    ('Poly(3,3,3-trifuoropropylene)', '*CC(*)C(F)(F)F', '{[$]CC(C(F)(F)F)[$]}', 300),
    ('Poly(vinyl acetate)', '*CC(*)OC(C)=O', '{[$]CC(OC(C)=O)[$]}', 301),
    ('Poly(4-methyl-1-pentene)', '*CC(*)CC(C)C', '{[$]CC(CC(C)C)[$]}', 302),
    ('Poly(vinyl formate)', '*CC(*)OC=O', '{[$]CC(OC=O)[$]}', 304),
    ('Poly(vinyl chloroacetate)', '*CC(*)OC(=O)CCl', '{[$]CC(OC(=O)CCl)[$]}', 304),
    ('Poly(neopentyl methacrylate)', '*CC(*)(C)C(=O)OCC(C)(C)C', '{[$]CC(C(=O)OCC(C)(C)C)(C)[$]}', 306),
    ('Poly(n-propyl methacrylate)', '*CC(*)(C)C(=O)OCCC', '{[$]CC(C(=O)OCCC)(C)[$]}', 308),
    ('Poly(12-aminododecanoic acid)', '*NCCCCCCCCCCCC(*)=O', '{[<]NCCCCCCCCCCCC(=O)[>]}', 310),
    ('Poly[di(p-tolyl)silylenetrimethylene]', '*CCC[Si](*)(c1ccc(C)cc1)c1ccc(C)cc1', '{[<]CCC[Si](c1ccc(C)cc1)(c1ccc(C)cc1)[>]}', 311),
    ('Poly(hexamethylene sebacamide)', '*NCCCCCCNC(=O)CCCCCCCCC(*)=O', '{[<]C(=O)CCCCCCCCC(=O)[<],[>]NCCCCCCN[>]}', 313),
    ('Poly(4-cyclohexyl-1-butene)', '*CC(*)CCC1CCCCC1', '{[$]CC(CCC1CCCCC1)[$]}', 313),
    ('Poly[(pentafluoroethyl)ethylene]', '*CC(*)C(F)(F)C(F)(F)F', '{[$]CC(C(F)(F)C(F)(F)F)[$]}', 314),
    ('Poly(11-aminoundecanoic acid)', '*NCCCCCCCCCCC(*)=O', '{[<]NCCCCCCCCCCC(=O)[>]}', 315),
    ('Poly(t-butyl acrylate)', '*CC(*)C(=O)OC(C)(C)C', '{[$]CC(C(=O)OC(C)(C)C)[$]}', 315),
    ('Poly(3-phenoxypropylene oxide)', '*CC(COc1ccccc1)O*', '{[<]CC(COc1ccccc1)O[>],[<]C(COc1ccccc1)CO[>]}', 315),
    ('Poly(2,3,3,3-tetrafluoropropylene)', '*CC(*)(F)C(F)(F)F', '{[$]CC(C(F)(F)F)(F)[$]}', 315),
    ('Poly(10-aminodecanoic acid)', '*NCCCCCCCCCC(*)=O', '{[<]NCCCCCCCCCC(=O)[>]}', 316),
    ('Poly[oxy(m-phenylene)]', '*Oc1cccc(*)c1', '{[<]Oc1cccc(c1)[>]}', 318),
    ('Poly(3,3-dimethylbutyl methacrylate)', '*CC(*)(C)C(=O)OCCC(C)(C)C', '{[$]CC(C(=O)OCCC(C)(C)C)(C)[$]}', 318),
    ('Poly(decamethylene sebacamide)', '*NCCCCCCCCCCNC(=O)CCCCCCCCC(*)=O', '{[<]C(=O)CCCCCCCCC(=O)[<],[>]NCCCCCCCCCCN[>]}', 319),
    ('Poly(n-butyl acrylamide)', '*CC(*)C(=O)NCCCC', '{[$]CC(C(=O)NCCCC)[$]}', 319),
    ('Poly(vinyl trifluoroacetate)', '*CC(*)OC(=O)C(F)(F)F', '{[$]CC(OC(=O)C(F)(F)F)[$]}', 319),
    ('Poly(p-n-butoxy styrene)', '*CC(*)c1ccc(OCCCC)cc1', '{[$]CC(c1ccc(OCCCC)cc1)[$]}', 320),
    ('Poly(isobutyl methacrylate)', '*CC(*)(C)C(=O)OCC(C)C', '{[$]CC(C(=O)OCC(C)C)(C)[$]}', 321),
    ('Poly(3-methyl-1-butene)', '*CC(*)C(C)C', '{[$]CC(C(C)C)[$]}', 323),
    ('Poly(9-aminononanoic acid)', '*NCCCCCCCCC(*)=O', '{[<]NCCCCCCCCC(=O)[>]}', 324),
    ('Poly(8-aminocaprylic acid)', '*NCCCCCCCC(*)=O', '{[<]NCCCCCCCC(=O)[>]}', 324),
    ('Poly(vinyl butyral)', '*CC1CC(*)OC(CCC)O1', '{[<]CC1CC(OC(CCC)O1)[>]}', 324),
    ('Poly(ethylene isophthalate)', '*OCCOC(=O)c1cccc(C(*)=O)c1', '{[<]C(=O)c1cccc(c1)C(=O)[<],[>]OCCO[>]}', 324),
    ('Poly(ethyl methacrylate)', '*CC(*)(C)C(=O)OCC', '{[$]CC(C(=O)OCC)(C)[$]}', 324),
    ('Poly[(4-dimethylaminophenyl) phenylsilylenetrimethylene)]', '*CCC[Si](*)(c1ccccc1)c1ccc(N(C)C)cc1', '{[<]CCC[Si](c1ccc(N(C)C)cc1)(c1ccccc1)[>]}', 325),
    ('Poly(isopropyl methacrylate)', '*CC(*)(C)C(=O)OC(C)C', '{[$]CC(C(=O)OC(C)C)(C)[$]}', 327),
    ('Poly(methyl-p-xylylene)', '*Cc1ccc(C*)c(C)c1', '{[$]Cc1ccc(c(C)c1)C[$]}', 328),
    ('Poly(vinyl isobutylral)', '*CC(*)C(=O)C(C)C', '{[<]CC1CC(OC(C(C)C)O1)[>]}', 329),
    ('Poly(n-butyl ??-chloroacrylate)', '*CC(*)(Cl)C(=O)OCCCC', '{[$]CC(C(=O)OCCCC)(Cl)[$]}', 330),
    ('Poly(7-aminoheptanoic acid)', '*NCCCCCCC(*)=O', '{[<]NCCCCCCC(=O)[>]}', 330),
    ('Poly(sec-butyl methacrylate)', '*CC(*)(C)C(=O)OC(C)CC', '{[$]CC(C(=O)OC(C)CC)(C)[$]}', 330),
    ('Poly(hexamethylene adipamide)', '*NCCCCCCNC(=O)CCCCC(*)=O', '{[<]C(=O)CCCCC(=O)[<],[>]NCCCCCCN[>]}', 330),
    ('Poly(p-isopentoxy styrene)', '*CC(*)c1ccc(OCCC(C)C)cc1', '{[$]CC(c1ccc(OCCC(C)C)cc1)[$]}', 330),
    ('Poly[(heptafluoropropyl)ethylene]', '*CC(*)C(F)(F)C(F)(F)C(F)(F)F', '{[$]CC(C(F)(F)C(F)(F)C(F)(F)F)[$]}', 331),
    ('Poly(oxydiphenylsilylene-1,3- phenylene)', '*O[Si](c1ccccc1)(c1ccccc1)c1cccc(*)c1', '{[<]O[Si](c1ccccc1)(c1ccccc1)c1cccc(c1)[>]}', 331),
    ('Poly(p-xylylene)', '*Cc1ccc(C*)cc1', '{[<]Cc1ccc(cc1)C[>]}', 333),
    ('Poly(3-cyclopentyl-1-propene)', '*CC(*)CC1CCCC1', '{[$]CC(CC1CCCC1)[$]}', 333),
    ('Poly(3-phenyl-1-propene)', '*CC(*)Cc1ccccc1', '{[$]CC(Cc1ccccc1)[$]}', 333),
    ('Poly(??-caprolactam)', '*NCCCCCC(*)=O', '{[<]NCCCCCC(=O)[>]}', 335),
    ('Poly(ethylene-1,4- naphthalenedicarboxylate)', '*OCCOC(=O)c1ccc(C(*)=O)c2ccccc12', '{[<]C(=O)c1ccc(c2ccccc12)C(=O)[<],[>]OCCO[>]}', 337),
    ('Poly(p-n-propoxy styrene)', '*CC(*)c1ccc(OCCC)cc1', '{[$]CC(c1ccc(OCCC)cc1)[$]}', 343),
    ('Poly(n-propyl ??-chloroacrylate)', '*CC(*)(Cl)C(=O)OCCC', '{[$]CC(C(=O)OCCC)(Cl)[$]}', 344),
    ('Poly(ethylene-1,5- naphthalenedicarboxylate)', '*OCCOC(=O)c1cccc2c(C(*)=O)cccc12', '{[<]C(=O)c1cccc2c(cccc12)C(=O)[<],[>]OCCO[>]}', 344),
    ('Poly(ethylene terephthalate)', '*OCCOC(=O)c1ccc(C(*)=O)cc1', '{[<]C(=O)c1ccc(cc1)C(=O)[<],[>]OCCO[>]}', 345),
    ('Poly(sec-butyl ??-chloroacrylate)', '*CC(*)(Cl)C(=O)OC(C)CC', '{[$]CC(C(=O)OC(C)CC)(Cl)[$]}', 347),
    ('Poly(vinyl chloride)', '*CC(*)Cl', '{[$]CC(Cl)[$]}', 348),
    ('Poly(3-cyclohexyl-1-propene)', '*CC(*)CC1CCCCC1', '{[$]CC(CC1CCCCC1)[$]}', 348),
    ('Poly(vinyl cyclopentane) ', '*CC(*)C1CCCC1', '{[$]CC(C1CCCC1)[$]}', 348),
    ('Poly(2-hydroxypropyl methacrylate)', '*CC(*)(C)C(=O)OCC(C)O', '{[$]CC(C(=O)OCC(C)O)(C)[$]}', 349),
    ('Poly(p-methoxymethyl styrene)', '*CC(*)c1ccc(COC)cc1', '{[$]CC(c1ccc(COC)cc1)[$]}', 350),
    ('Poly(chloro-p-xylylene)', '*Cc1ccc(C*)c(Cl)c1', '{[$]Cc1ccc(c(Cl)c1)C[$]}', 353),
    ('Poly(bromo-p-xylylene)', '*Cc1ccc(C*)c(Br)c1', '{[$]Cc1ccc(c(Br)c1)C[$]}', 353),
    ('Poly(ethylene oxybenzoate)', '*CCOC(=O)c1ccc(O*)cc1', '{[<]CCOC(=O)c1ccc(cc1)O[>]}', 355),
    ('Poly(vinyl alcohol)', '*CC(*)O', '{[$]CC(O)[$]}', 358),
    ('Poly[oxy(p-phenylene)]', '*Oc1ccc(*)cc1', '{[<]Oc1ccc(cc1)[>]}', 358),
    ('Poly(p-sec-butyl styrene)', '*CC(*)c1ccc(C(C)CC)cc1', '{[$]CC(c1ccc(C(C)CC)cc1)[$]}', 359),
    ('Poly(p-ethoxy styrene)', '*CC(*)c1ccc(OCC)cc1', '{[$]CC(c1ccc(OCC)cc1)[$]}', 359),
    ('Poly(2-hydroxyethyl methacrylate)', '*CC(*)(C)C(=O)OCCO', '{[$]CC(C(=O)OCCO)(C)[$]}', 359),
    ('Poly[thio(p-phenylene)]', '*Sc1ccc(*)cc1', '{[<]Sc1ccc(cc1)[>]}', 360),
    ('Poly(p-isopropyl styrene)', '*CC(*)c1ccc(C(C)C)cc1', '{[$]CC(c1ccc(C(C)C)cc1)[$]}', 360),
    ('Poly(2-methyl-5-t-butyl styrene)', '*CC(*)c1cc(C(C)(C)C)ccc1C', '{[$]CC(c1cc(C(C)(C)C)ccc1C)[$]}', 360),
    ('Poly(p-methoxy styrene)', '*CC(*)c1ccc(OC)cc1', '{[$]CC(c1ccc(OC)cc1)[$]}', 362),
    ('Poly(isopropyl ??-chloroacrylate)', '*CC(*)(Cl)C(=O)OC(C)C', '{[$]CC(C(=O)OC(C)C)(Cl)[$]}', 363),
    ('Poly(4-methoxy-2-methyl styrene)', '*CC(*)c1ccc(OC)cc1C', '{[$]CC(c1ccc(OC)cc1C)[$]}', 363),
    ('Poly(vinyl cyclohexane)', '*CC(*)C1CCCCC1', '{[$]CC(C1CCCCC1)[$]}', 363),
    ('Poly(cyano-p-xylylene)', '*Cc1ccc(C*)c(C#N)c1', '{[$]Cc1ccc(c(C#N)c1)C[$]}', 363),
    ("Poly(??,??,??',??'-tetratluoro-p-xylylene)", '*c1ccc(C(F)(F)C(*)(F)F)cc1', '{[<]c1ccc(cc1)[<],[>]C(F)(F)C(F)(F)[>]}', 363),
    ('Poly(m-xylylene adipamide)', '*NCc1cccc(CNC(=O)CCCCC(*)=O)c1', '{[<]NCc1cccc(c1)CNC(=O)CCCCC(=O)[>]}', 363),
    ('Poly(m-chloro styrene)', '*CC(*)c1cccc(Cl)c1', '{[$]CC(c1cccc(Cl)c1)[$]}', 363),
    ('Poly(2-chloroethyl methacrylate)', '*CC(*)(C)C(=O)OCCCl', '{[$]CC(C(=O)OCCCl)(C)[$]}', 365),
    ('Poly(ethyl ??-chloroacrylate)', '*CC(*)(Cl)C(=O)OCC', '{[$]CC(C(=O)OCC)(Cl)[$]}', 366),
    ('Poly(cyclohexylmethylsilane)', '*[Si](*)(C)C1CCCCC1', '{[$][Si](C1CCCCC1)(C)[$]}', 366),
    ('Poly(1,4-cyclohexylidene dimethylene terephthalate)', '*OCC1CCC(COC(=O)c2ccc(C(*)=O)cc2)CC1', '{[<]C(=O)c2ccc(cc2)C(=O)[<],[>]OCC1CCC(CC1)CO[>]}', 368),
    ('Poly(m-methyl styrene)', '*CC(*)c1cccc(C)c1', '{[$]CC(c1cccc(C)c1)[$]}', 370),
    ('Poly(2,5-dimethyl-p-xylylene)', '*Cc1cc(C)c(C*)cc1C', '{[<]Cc1cc(C)c(cc1C)C[>]}', 373),
    ('Polychlorotrifluoroethylene', '*C(F)(F)C(*)(F)Cl', '{[$]C(F)(F)C(Cl)(F)[$]}', 373),
    ('Polystyrene', '*CC(*)c1ccccc1', '{[$]CC(c1ccccc1)[$]}', 373),
    ('Poly(p-methyl styrene)', '*CC(*)c1ccc(C)cc1', '{[$]CC(c1ccc(C)cc1)[$]}', 374),
    ('Poly(2,5-difluoro styrene)', '*CC(*)c1cc(F)ccc1F', '{[$]CC(c1cc(F)ccc1F)[$]}', 374),
    ('Poly(o-ethyl styrene)', '*CC(*)c1ccccc1CC', '{[$]CC(c1ccccc1CC)[$]}', 376),
    ('Poly(3,5-dimethyl styrene)', '*CC(*)c1cc(C)cc(C)c1', '{[$]CC(c1cc(C)cc(C)c1)[$]}', 377),
    ('Poly(cyclohexyl methacrylate)', '*CC(*)(C)C(=O)OC1CCCCC1', '{[$]CC(C(=O)OC1CCCCC1)(C)[$]}', 377),
    ('Poly(o-vinyl pyridine)', '*CC(*)c1ccccn1', '{[$]CC(c1ccccn1)[$]}', 377),
    ('Poly(methyl methacrylate)', '*CC(*)(C)C(=O)OC', '{[$]CC(C(=O)OC)(C)[$]}', 378),
    ('Polyacrylonitrile', '*CC(*)C#N', '{[$]CC(C#N)[$]}', 378),
    ('Poly(vinyl formal)', '*CC1CC(*)OCO1', '{[<]CC1CC(OCO1)[>]}', 378),
    ('Poly(o-fluoro styrene)', '*CC(*)c1ccccc1F', '{[$]CC(c1ccccc1F)[$]}', 378),
    ('Poly(2,3,4,5,6-pentafluoro styrene)', '*CC(*)c1c(F)c(F)c(F)c(F)c1F', '{[$]CC(c1c(F)c(F)c(F)c(F)c1F)[$]}', 378),
    ('Poly(acrylic acid)', '*CC(*)C(=O)O', '{[$]CC(C(=O)O)[$]}', 379),
    ('Poly(p-fluoro styrere)', '*CC(*)c1ccc(F)cc1', '{[$]CC(c1ccc(F)cc1)[$]}', 379),
    ('Poly(t-butyl methacrylate)', '*CC(*)(C)C(=O)OC(C)(C)C', '{[$]CC(C(=O)OC(C)(C)C)(C)[$]}', 380),
    ('Poly(3,4-dimethyl styrene)', '*CC(*)c1ccc(C)c(C)c1', '{[$]CC(c1ccc(C)c(C)c1)[$]}', 384),
    ('Poly(2-fluoro-5-methyl styrene)', '*CC(*)c1cc(C)ccc1F', '{[$]CC(c1cc(C)ccc1F)[$]}', 384),
    ('Poly(2,4 dimethyl styrene)', '*CC(*)c1ccc(Cl)cc1Cl', '{[$]CC(c1ccc(C)cc1C)[$]}', 385),
    ('Poly(p-methoxycarbonyl styrene)', '*CC(*)c1ccc(C(=O)OC)cc1', '{[$]CC(c1ccc(C(=O)OC)cc1)[$]}', 386),
    ('Poly(3-methyl-4-chloro styrene)', '*CC(*)c1ccc(Cl)c(C)c1', '{[$]CC(c1ccc(Cl)c(C)c1)[$]}', 387),
    ('Poly(cyclohexyl ??-chloroacrylate)', '*CC(*)(Cl)C(=O)OC1CCCCC1', '{[$]CC(C(=O)OC1CCCCC1)(Cl)[$]}', 387),
    ('Poly(p-xylylene sebacamide)', '*NCc1ccc(CNC(=O)CCCCCCCCC(*)=O)cc1', '{[<]C(=O)CCCCCCCCC(=O)[<],[>]NCc1ccc(cc1)CN[>]}', 388),
    ('Poly[thio bis(4-phenyl)carbonate]', '*Oc1ccc(Sc2ccc(OC(*)=O)cc2)cc1', '{[<]Oc1ccc(cc1)Sc2ccc(cc2)O[<],[>]C(=O)[>]}', 388),
    ('Poly(p-chloro styrene)', '*CC(*)c1ccc(Cl)cc1', '{[$]CC(c1ccc(Cl)cc1)[$]}', 389),
    ('Perfluoropolymer', '*C(F)(F)C1(F)C(F)(F)C(*)(F)C(F)(F)C1(F)F', '{[<]c1nc(C(F)(F)C(F)(F)OC(F)(F)F)nc(n1)C(F)(F)C(F)(F)C(F)(F)C(F)(F)[>]}', 390),
    ('Poly(phenylmethylsilane)', '*[Si](*)(C)c1ccccc1', '{[$][Si](c1ccccc1)(C)[$]}', 390),
    ('Poly(o-chloro styrene)', '*CC(*)c1ccccc1Cl', '{[$]CC(c1ccccc1Cl)[$]}', 392),
    ('Poly[2,2-butane bis (4-(2-methylphenyl)) carbonate]', '*Oc1ccc(C(C)(CC)c2ccc(OC(*)=O)cc2C)c(C)c1', '{[<]Oc1ccc(cc1C)C(C)(CC)c2ccc(c(C)c2)O[<],[>]C(=O)[>]}', 392),
    ('Poly(2,5-dichloro styrene)', '*CC(*)c1cc(Cl)ccc1Cl', '{[$]CC(c1cc(Cl)ccc1Cl)[$]}', 393),
    ('Poly(phenyl methacrylate)', '*CC(*)(C)C(=O)Oc1ccccc1', '{[$]CC(C(=O)Oc1ccccc1)(C)[$]}', 393),
    ('Polymethacrylonitrile', '*CC(*)(C)C#N', '{[$]CC(C#N)(C)[$]}', 393),
    ('Poly(??-p-dimethyl styrene)', '*CC(*)(C)c1ccc(C)cc1', '{[$]CC(c1ccc(C)cc1)(C)[$]}', 394),
    ('Poly(3-fluoro-4-chloro styrene)', '*CC(*)c1ccc(Cl)c(F)c1', '{[$]CC(c1ccc(Cl)c(F)c1)[$]}', 395),
    ('Poly[1,1-butane bis(4- phenyl)carbonate]', '*Oc1ccc(C(CCC)c2ccc(OC(*)=O)cc2)cc1', '{[<]Oc1ccc(cc1)C(CCC)c2ccc(cc2)O[<],[>]C(=O)[>]}', 396),
    ('Poly(ethylene-2,6- naphthalenedicarboxylate)', '*OCCOC(=O)c1ccc2cc(C(*)=O)ccc2c1', '{[<]C(=O)c1ccc2cc(ccc2c1)C(=O)[<],[>]OCCO[>]}', 397),
    ('Poly(m-hydroxymethyl styrene)', '*CC(*)c1cccc(CO)c1', '{[$]CC(c1cccc(CO)c1)[$]}', 398),
    ('Poly(3,4-dichlorostyrene)', '*CC(*)c1ccc(Cl)c(Cl)c1', '{[$]CC(c1ccc(Cl)c(Cl)c1)[$]}', 401),
    ('Polyetherimide', '*Oc1ccc(Oc2ccc(Oc3ccc4c(c3)C(=O)N(CCCCCCN3C(=O)c5ccc(*)cc5C3=O)C4=O)cc2)cc1', '{[<]CCN1C(=O)c2ccc(cc2C1=O)Oc3ccc(cc3)Oc4ccc(cc4)Oc5ccc6c(c5)C(=O)N(C6=O)[>]}', 410),
    ('Poly(p-butyl styrene)', '*CC(*)c1ccc(C(C)(C)C)cc1', '{[$]CC(c1ccc(CCCC)cc1)[$]}', 402),
    ('Poly(hexamethylene isophthalamide)', '*NCCCCCCNC(=O)c1cccc(C(*)=O)c1', '{[<]C(=O)c1cccc(c1)C(=O)[<],[>]NCCCCCCN[>]}', 403),
    ('Poly[1,1-ethane bis(4- phenyl)carbonate]', '*Oc1ccc(C(C)c2ccc(OC(*)=O)cc2)cc1', '{[<]Oc1ccc(cc1)C(C)c2ccc(cc2)O[<],[>]C(=O)[>]}', 403),
    ('Poly(2,4-dichloro styrene)', '*CC(*)c1ccc(C)cc1C', '{[$]CC(c1ccc(Cl)cc1Cl)[$]}', 406),
    ('Poly[2,2 butane bis(4- phenyl)carbonate]', '*Oc1ccc(C(C)(CC)c2ccc(OC(*)=O)cc2)cc1', '{[<]Oc1ccc(cc1)C(C)(CC)c2ccc(cc2)O[<],[>]C(=O)[>]}', 407),
    ('Poly(o-methyl styrene)', '*CC(*)c1ccccc1C', '{[$]CC(c1ccccc1C)[$]}', 409),
    ('Poly(??-methyl styrene)', '*CC(*)(C)c1ccccc1', '{[$]CC(c1ccccc1)(C)[$]}', 409),
    ('Poly[2,2-pentane bis(4- phenyl)carbonate]', '*Oc1ccc(C(C)(CCC)c2ccc(OC(*)=O)cc2)cc1', '{[<]Oc1ccc(cc1)C(C)(CCC)c2ccc(cc2)O[<],[>]C(=O)[>]}', 410),
    ('Poly(m-phenylene isophthalate)', '*Oc1cccc(OC(=O)c2cccc(C(*)=O)c2)c1', '{[<]C(=O)c2cccc(c2)C(=O)[<],[>]Nc1cccc(c1)N[>]}', 411),
    ('Poly(p-phenyl styrene)', '*CC(*)c1ccc(-c2ccccc2)cc1', '{[$]CC(c1ccc(c2ccccc2)cc1)[$]}', 411),
    ('Poly(oxycarbonyloxy-2-chloro-1,4- phenyleneisopropylidene-2-methyl-1,4- phenylene)', '*Oc1ccc(C(C)(C)c2ccc(OC(*)=O)cc2)cc1Cl', '{[<]Oc1ccc(cc1C)C(C)(C)c2ccc(c(Cl)c2)O[<],[>]C(=O)[>]}', 411),
    ('Poly(p-hydroxymethyl styrene)', '*CC(*)c1ccc(CO)cc1', '{[$]CC(c1ccc(CO)cc1)[$]}', 413),
    ('Poly(p-vinyl pyridine)', '*CC(*)c1ccncc1', '{[$]CC(c1ccncc1)[$]}', 415),
    ('Poly[2,2-butane bis{4-(2- chlorophenyl)} carbonate]', '*OC(=O)Oc1ccc(C(C)(CC)c2ccc(*)cc2Cl)c(Cl)c1', '{[<]OC(=O)Oc1ccc(c(Cl)c1)C(C)(CC)c2ccc(cc2Cl)[>]}', 415),
    ('Poly(2,5-dimethyl styrene)', '*CC(*)c1cc(C)ccc1C', '{[$]CC(c1cc(C)ccc1C)[$]}', 416),
    ('Poly(p-bromo styrene)', '*CC(*)c1ccc(Br)cc1', '{[$]CC(c1ccc(Br)cc1)[$]}', 417),
    ('Poly(2-methyl-4-chloro styrene)', '*CC(*)c1ccc(Cl)cc1C', '{[$]CC(c1ccc(Cl)cc1C)[$]}', 418),
    ('Poly(N-vinyl pyrrolidone)', '*CC(*)N1CCCC1=O', '{[$]CC(N1CCCC1=O)[$]}', 418),
    ('Poly(oxycarbonyloxy-2-chloro-1,4- phenyleneisopropylidene-1,4- phenylene)', '*Oc1c(Cl)cc(C(C)(C)c2ccc(OC(*)=O)cc2)cc1Cl', '{[<]Oc1ccc(cc1Cl)C(C)(C)c2ccc(cc2)O[<],[>]C(=O)[>]}', 419),
    ('Poly[2,2-propane bis{4-(2- chlorophenyl)}carbonate]', '*OC(=O)Oc1ccc(C(C)(C)c2ccc(*)cc2Cl)c(Cl)c1', '{[<]Oc1ccc(cc1Cl)C(C)(C)c2ccc(c(Cl)c2)O[<],[>]C(=O)[>]}', 419),
    ('Poly(oxy-1,4-phenylene-oxy-1,4- phenylene-carbonyl-1,4-phenylene)', '*Oc1ccc(Oc2ccc(C(=O)c3ccc(*)cc3)cc2)cc1', '{[<]c2ccc(cc2)C(=O)c3ccc(cc3)[<],[>]Oc1ccc(cc1)O[>]}', 419),
    ('Poly[methane bis(4-phenyl)carbonate]', '*Oc1ccc(Cc2ccc(OC(*)=O)cc2)cc1', '{[<]Oc1ccc(cc1)Cc2ccc(cc2)O[<],[>]C(=O)[>]}', 420),
    ('Poly(p-hydroxybenzoate)', '*Oc1ccc(C(*)=O)cc1', '{[<]Oc1ccc(cc1)C(=O)[>]}', 420),
    ('Poly[4,4-heptane bis(4- phenyl)carbonate]', '*Oc1ccc(C(CCC)(CCC)c2ccc(OC(*)=O)cc2)cc1', '{[<]Oc1ccc(cc1)C(CCC)(CCC)c2ccc(cc2)O[<],[>]C(=O)[>]}', 421),
    ('Poly[1,1-(2-methyl propane) bis(4- phenyl)carbonate]', '*Oc1ccc(C(c2ccc(OC(*)=O)cc2)C(C)C)cc1', '{[<]Oc1ccc(cc1)C(C(C)C)c2ccc(cc2)O[<],[>]C(=O)[>]}', 422),
    ('Bisphenol-A polycarbonate', '*Oc1ccc(C(C)(C)c2ccc(OC(*)=O)cc2)cc1', '{[<]Oc1ccc(cc1)C(C)(C)c2ccc(cc2)O[<],[>]C(=O)[>]}', 423),
    ('Poly(N-vinyl carbazole)', '*CC(*)n1c2ccccc2c2ccccc21', '{[$]CC(n1c2ccccc2c2ccccc21)[$]}', 423),
    ('Poly(??-vinyl naphthalene)', '*CC(*)c1ccc2ccccc2c1', '{[$]CC(c1ccc2ccccc2c1)[$]}', 424),
    ('Polyhexafluoropropylene', '*C(F)(F)C(*)(F)C(F)(F)F', '{[$]C(F)(F)C(C(F)(F)F)(F)[$]}', 425),
    ('Poly[1,1-dichloroethylene bis(4- phenyl)carbonate]', '*OC(=O)Oc1ccc(C(=C(Cl)Cl)c2ccc(*)cc2)cc1', '{[<]OC(=O)Oc1ccc(cc1)C(=C(Cl)Cl)c2ccc(cc2)[>]}', 430),
    ('Poly(??-vinyl naphthalene)', '*CC(*)c1cccc2ccccc12', '{[$]CC(c1cccc2ccccc12)[$]}', 432),
    ('Poly(o-hydroxymethyl styrene)', '*CC(*)c1ccccc1CO', '{[$]CC(c1ccccc1CO)[$]}', 433),
    ('Poly(methyl ??-cyanoacrylate)', '*CC(*)(C#N)C(=O)OC', '{[$]CC(C(=O)OC)(C#N)[$]}', 433),
    ('Poly[1,1-cyclopentane bis(4- phenyl)carbonate]', '*Oc1ccc(C2(c3ccc(OC(*)=O)cc3)CCCC2)cc1', '{[<]Oc1ccc(cc1)C2(CCCC2)c3ccc(cc3)O[<],[>]C(=O)[>]}', 440),
    ('Poly(oxyterephthaloyloxy-2-methyl-1,4-phenyleneisopropylidene-3-methyl-1,4-phenylene)', '*Oc1ccc(C(C)(C)c2ccc(OC(=O)c3ccc(C(*)=O)cc3)c(C)c2)cc1C', '{[<]C(=O)c1ccc(cc1)C(=O)[<],[>]Oc2ccc(cc2C)C(C)(C)c3ccc(c(C)c3)O[>]}', 444),
    ('Poly[1,1-cyclohexane bis(4- phenyl)carbonate]', '*Oc1ccc(C2(c3ccc(OC(*)=O)cc3)CCCCC2)cc1', '{[<]Oc1ccc(cc1)C2(CCCCC2)c3ccc(cc3)O[<],[>]C(=O)[>]}', 444),
    ('Poly[2,2-hexafluoropropane bis(4- phenyl)carbonate]', '*Oc1ccc(C(c2ccc(OC(*)=O)cc2)(C(F)(F)F)C(F)(F)F)cc1', '{[<]Oc1ccc(cc1)C(C(F)(F)F)(C(F)(F)F)c2ccc(cc2)O[<],[>]C(=O)[>]}', 449),
    ('Poly[1,1-(1-phenylethane) bis(4- phenyl)carbonate]', '*Oc1ccc(C(C)(c2ccccc2)c2ccc(OC(*)=O)cc2)cc1', '{[<]Oc1ccc(cc1)C(C)(c2ccccc2)c2ccc(cc2)O[<],[>]C(=O)[>]}', 449),
    ('Poly[2,2-(1,3-dichloro-1,1,3,3- tetrafluoro)propane bis(4- phenyl)carbonate]', '*Oc1ccc(C(c2ccc(OC(*)=O)cc2)(C(F)(F)Cl)C(F)(F)Cl)cc1', '{[<]Oc1ccc(cc1)C(C(F)(F)Cl)(C(F)(F)Cl)c2ccc(cc2)O[<],[>]C(=O)[>]}', 457),
    ("Poly[4,4'-isopropylidene diphenoxy di(4-phenylene) sulfone]", '*Oc1ccc(C(C)(C)c2ccc(Oc3ccc(S(=O)(=O)c4ccc(*)cc4)cc3)cc2)cc1', '{[<]Oc3ccc(cc3)C(C)(C)c4ccc(cc4)O[<],[>]c1ccc(cc1)S(=O)(=O)c2ccc(cc2)[>]}', 458),
    ('Poly(oxycarbonyloxy-2,6-dichloro-1,4- phenyleneisopropylidene-1,4- phenylene)', '*CCC(C)CCC(=O)O*', '{[<]Oc1ccc(cc1)C(C)(C)c2cc(Cl)c(c(Cl)c2)O[<],[>]C(=O)[>]}', 459),
    ('Poly(perfluorostyrene)', '*C(F)(F)C(*)(F)c1c(F)c(F)c(F)c(F)c1F', '{[$]C(F)(F)C(c1c(F)c(F)c(F)c(F)c1F)(F)[$]}', 467),
    ('Poly[2,2-propane bis(4-(2,6- dimethylphenyl)} carbonate]', '*Oc1cc(C)c(C(C)(C)c2c(C)cc(OC(*)=O)cc2C)c(C)c1', '{[<]Oc1cc(C)c(c(C)c1)C(C)(C)c2c(C)cc(cc2C)O[<],[>]C(=O)[>]}', 473),
    ('Polyetherimide', '*c1cccc(C(=O)c2cccc(N3C(=O)c4ccc(Oc5ccc(Sc6ccc(Oc7ccc8c(c7)C(=O)N(*)C8=O)cc6)cc5)cc4C3=O)c2)c1', '{[<]c1cccc(c1)C(=O)c2cccc(c2)[<],[>]N3C(=O)c4ccc(cc4C3=O)Oc5ccc(cc5)Sc6ccc(cc6)Oc7ccc8c(c7)C(=O)N(C8=O)[>]}', 473),
    ('Poly(??,??,??-trifluoro styrene)', '*C(F)(F)C(*)(F)c1ccccc1', '{[$]C(F)(F)C(c1ccccc1)(F)[$]}', 475),
    ('Poly(bisphenol-A terephthalate)', '*Oc1ccc(C(C)(C)c2ccc(OC(=O)c3ccc(C(*)=O)cc3)cc2)cc1', '{[<]C(=O)c1ccc(cc1)C(=O)[<],[>]Oc2ccc(cc2)C(C)(C)c3ccc(cc3)O[>]}', 478),
    ('Poly[oxy(2,6-dimethyl-1,4-phenylene)]', '*Oc1c(C)cc(*)cc1C', '{[<]Oc1c(C)cc(cc1C)[>]}', 482),
    ('Polyetherimide', '*c1cccc(N2C(=O)c3ccc(Oc4ccc(Sc5ccc(Oc6ccc7c(c6)C(=O)N(*)C7=O)cc5)cc4)cc3C2=O)c1', '{[<]c3cccc(c3)[<],[>]N5C(=O)c6ccc(cc6C5=O)Oc1ccc(cc1)Sc2ccc(cc2)Oc3ccc4c(c3)C(=O)N(C4=O)[>]}', 482),
    ('Polyetherimide', '*c1ccc(C(=O)c2ccccc2N2C(=O)c3ccc(Oc4ccc(Sc5ccc(Oc6ccc7c(c6)C(=O)N(*)C7=O)cc5)cc4)cc3C2=O)cc1', '{[<]c1ccc(cc1)C(=O)c2ccccc2[<],[>]N2C(=O)c3ccc(cc3C2=O)Oc4ccc(cc4)Sc5ccc(cc5)Oc6ccc7c(c6)C(=O)N(C7=O)[>]}', 485),
    ('Polyetherimide', '*c1ccc(C(=O)c2cccc(N3C(=O)c4ccc(Oc5ccc(Sc6ccc(Oc7ccc8c(c7)C(=O)N(*)C8=O)cc6)cc5)cc4C3=O)c2)cc1', '{[<]c1ccc(cc1)C(=O)c2cccc(c2)[<],[>]N3C(=O)c4ccc(cc4C3=O)Oc5ccc(cc5)Sc6ccc(cc6)Oc7ccc8c(c7)C(=O)N(C8=O)[>]}', 486),
    ('Poly[oxy(2,6-diphenyl-1,4-phenylene)]', '*Oc1c(-c2ccccc2)cc(*)cc1-c1ccccc1', '{[<]Oc1c(c2ccccc2)cc(cc1c1ccccc1)[>]}', 493),
    ("Poly[4,4'-diphenoxy di(4- phenylene)sulfone]", '*Oc1ccc(-c2ccc(Oc3ccc(S(=O)(=O)c4ccc(*)cc4)cc3)cc2)cc1', '{[<]Oc2ccc(cc2)S(=O)(=O)c3ccc(cc3)O[<],[>]c1ccc(cc1)[>]}', 493),
    ("Poly[4,4'-sulfone diphenoxy di(4- phenylene)sulfone]", '*Oc1ccc(S(=O)(=O)c2ccc(Oc3ccc(S(=O)(=O)c4ccc(*)cc4)cc3)cc2)cc1', '{[<]Oc1ccc(cc1)S(=O)(=O)c2ccc(cc2)O[<],[>]c3ccc(cc3)S(=O)(=O)c4ccc(cc4)[>]}', 493),
    ('Ultem', '*c1cccc(N2C(=O)c3ccc(Oc4ccc(C(C)(C)c5ccc(Oc6ccc7c(c6)C(=O)N(*)C7=O)cc5)cc4)cc3C2=O)c1', '{[<]c3cccc(c3)[<],[>]N5C(=O)c6ccc(cc6C5=O)Oc1ccc(cc1)C(C)(C)c2ccc(cc2)Oc3ccc4c(c3)C(=O)N(C4=O)[>]}', 493),
    ('Polyetherimide', '*c1ccc(C(=O)c2ccc(N3C(=O)c4ccc(Oc5ccc(Sc6ccc(Oc7ccc8c(c7)C(=O)N(*)C8=O)cc6)cc5)cc4C3=O)cc2)cc1', '{[<]c1ccc(cc1)C(=O)c2ccc(cc2)[<],[>]N3C(=O)c4ccc(cc4C3=O)Oc5ccc(cc5)Sc6ccc(cc6)Oc7ccc8c(c7)C(=O)N(C8=O)[>]}', 494),
    ("Poly[N,N'-(m,m'-oxydiphenylene-oxy-m-phenylene) pyromellitimide]", '*Oc1cccc(Oc2cccc(-n3c(=O)c4cc5c(=O)n(-c6cccc(*)c6)c(=O)c5cc4c3=O)c2)c1', '{[<]Oc1cccc(c1)Oc2cccc(c2)n3c(=O)c4cc5c(=O)n(c(=O)c5cc4c3=O)c6cccc(c6)[>]}', 494),
    ('Poly(oxyterephthaloyloxy-2,6- dimethyl-1,4-phenylene\x02isopropylidene-3,5-dimethyl-1,4- phenylene)', '*Oc1c(C)cc(C(C)(C)c2cc(C)c(OC(=O)c3ccc(C(*)=O)cc3)c(C)c2)cc1C', '{[<]C(=O)c3ccc(cc3)C(=O)[<],[>]Oc1c(C)cc(cc1C)C(C)(C)c2cc(C)c(c(C)c2)O[>]}', 498),
    ('Polyetherimide', '*c1cccc(N2C(=O)c3ccc(Oc4ccc(Oc5ccc(Oc6ccc7c(c6)C(=O)N(*)C7=O)cc5)cc4)cc3C2=O)c1', '{[<]c3cccc(c3)[<],[>]N5C(=O)c6ccc(cc6C5=O)Oc1ccc(cc1)Oc2ccc(cc2)Oc3ccc4c(c3)C(=O)N(C4=O)[>]}', 500),
    ('Poly[2,2-propane bis{4-(2,6- dichlorophenyl)} carbonate]', '*Oc1c(Cl)cc(C(C)(C)c2cc(Cl)c(OC(*)=O)c(Cl)c2)cc1Cl', '{[<]Oc1c(Cl)cc(cc1Cl)C(C)(C)c2cc(Cl)c(c(Cl)c2)O[<],[>]C(=O)[>]}', 503),
    ('Polycarbonate', '*Oc1ccc(C2(c3ccc(OC(*)=O)cc3)CC3CCC2C3)cc1', '{[<]Oc1ccc(cc1)C2(CC3CCC2C3)c3ccc(cc3)O[<],[>]C(=O)[>]}', 505),
    ('Polyetherimide', '*Oc1ccc(C(=O)c2ccc(Oc3ccc4c(c3)C(=O)N(c3cccc(N5C(=O)c6ccc(*)cc6C5=O)c3)C4=O)cc2)cc1', '{[<]Oc1ccc(cc1)C(=O)c2ccc(cc2)Oc3ccc4c(c3)C(=O)N(C4=O)c3cccc(c3)N5C(=O)c6ccc(cc6C5=O)[>]}', 512),
    ('Polyimide', '*c1cccc(C(=O)c2cccc(N3C(=O)c4ccc(C(=O)c5ccc6c(c5)C(=O)N(*)C6=O)cc4C3=O)c2)c1', '{[<]c1cccc(c1)C(=O)c2cccc(c2)N3C(=O)c4ccc(cc4C3=O)C(=O)c5ccc6c(c5)C(=O)N(C6=O)[>]}', 513),
    ('Polycarbonate', '*Oc1ccc(C2(c3ccc(OC(*)=O)cc3)CC3CC2C2CCCC32)cc1', '{[<]Oc1ccc(cc1)C2(CC3CC2C2CCCC32)c3ccc(cc3)O[<],[>]C(=O)[>]}', 520),
    ('Polyetherimide', '*c1cccc(N2C(=O)c3ccc(Oc4ccc(-c5ccc(Oc6ccc7c(c6)C(=O)N(*)C7=O)cc5)cc4)cc3C2=O)c1', '{[<]c3cccc(c3)[<],[>]N5C(=O)c6ccc(cc6C5=O)Oc1ccc(cc1)c2ccc(cc2)Oc3ccc4c(c3)C(=O)N(C4=O)[>]}', 520),
    ('Poly[2,2-propane bis{4-(2,6- dibromophenyl)} carbonate]', '*Oc1c(Br)cc(C(C)(C)c2cc(Br)c(OC(*)=O)c(Br)c2)cc1Br', '{[<]Oc1c(Br)cc(cc1Br)C(C)(C)c2cc(Br)c(c(Br)c2)O[<],[>]C(=O)[>]}', 523),
    ('Polyimide', '*c1cccc(C(=O)c2cccc(N3C(=O)c4ccc(C(c5ccc6c(c5)C(=O)N(*)C6=O)(C(F)(F)F)C(F)(F)F)cc4C3=O)c2)c1', '{[<]c1cccc(c1)C(=O)c2cccc(c2)[<],[>]N3C(=O)c4ccc(cc4C3=O)C(C(F)(F)F)(C(F)(F)F)c5ccc6c(c5)C(=O)N(C6=O)[>]}', 533),
    ('Polyquinoline', '*c1ccc(Oc2ccc(-c3cc(-c4ccccc4)c4cc(Oc5ccc6nc(*)cc(-c7ccccc7)c6c5)ccc4n3)cc2)cc1', '{[<]c1ccc(cc1)Oc2ccc(cc2)c3cc(c4ccccc4)c4cc(ccc4n3)Oc5ccc6nc(cc(c7ccccc7)c6c5)[>]}', 539),
    ('Polyquinoline', '*c1cccc(-c2cc(-c3ccc(Oc4ccc(-c5cc(*)nc6ccccc56)cc4)cc3)c3ccccc3n2)c1', '{[<]c1cccc(c1)[<],[>]c2cc(c3ccccc3n2)c3ccc(cc3)Oc4ccc(cc4)c5cc(nc6ccccc56)[>]}', 541),
    ('Polyquinoline', '*c1cccc(-c2nc3ccccc3c(-c3ccc(Oc4ccc(-c5c(-c6ccccc6)c(*)nc6ccccc56)cc4)cc3)c2-c2ccccc2)c1', '{[<]c1cccc(c1)[<],[>]c2nc3ccccc3c(c2c2ccccc2)c3ccc(cc3)Oc4ccc(cc4)c5c(c6ccccc6)c(nc6ccccc56)[>]}', 546),
    ('Poly[o-biphenylenemethane bis(4- phenyl)carbonate]', '*Oc1ccc(-c2ccccc2Cc2ccccc2-c2ccc(OC(*)=O)cc2)cc1', '{[<]Oc1ccc(cc1)c2ccccc2Cc2ccccc2c2ccc(cc2)O[<],[>]C(=O)[>]}', 548),
    ('Polyimide', '*c1ccc(C(=O)c2cccc(N3C(=O)c4ccc(C(c5ccc6c(c5)C(=O)N(*)C6=O)(C(F)(F)F)C(F)(F)F)cc4C3=O)c2)cc1', '{[<]c1ccc(cc1)C(=O)c2cccc(c2)[<],[>]N3C(=O)c4ccc(cc4C3=O)C(C(F)(F)F)(C(F)(F)F)c5ccc6c(c5)C(=O)N(C6=O)[>]}', 561),
    ('Polyimide', '*c1ccc(C(=O)c2ccccc2N2C(=O)c3ccc(C(c4ccc5c(c4)C(=O)N(*)C5=O)(C(F)(F)F)C(F)(F)F)cc3C2=O)cc1', '{[<]c1ccc(cc1)C(=O)c2ccccc2[<],[>]N2C(=O)c3ccc(cc3C2=O)C(C(F)(F)F)(C(F)(F)F)c4ccc5c(c4)C(=O)N(C5=O)[>]}', 562),
    ('Polyquinoline', '*c1ccc(-c2cc(-c3ccccc3)c3cc(Oc4ccc5nc(*)cc(-c6ccccc6)c5c4)ccc3n2)cc1', '{[<]c1ccc(cc1)c2cc(c3ccccc3)c3cc(ccc3n2)Oc4ccc5nc(cc(c6ccccc6)c5c4)[>]}', 573),
    ("Poly(quinoxaline-2,7-diylquinoxaline-7,2-diyl-p-terphenyl-4,4'-ylene)", '*c1ccc(-c2ccc(-c3ccc(-c4cnc5ccc(-c6ccc7ncc(*)nc7c6)cc5n4)cc3)cc2)cc1', '{[<]c1ccc(cc1)c2ccc(cc2)c3ccc(cc3)c4cnc5ccc(cc5n4)c6ccc7ncc(nc7c6)[>]}', 578),
    ('Poly(quinoxaline-2, 7- diyloxyquinoxaline-7,2-diyl-1,4- phenylene)', '*c1ccc(-c2cnc3ccc(-c4ccc5ncc(*)nc5c4)cc3n2)cc1', '{[<]c1ccc(cc1)c2cnc3ccc(cc3n2)Oc4ccc5ncc(nc5c4)[>]}', 578),
    ('Polyphenolphthalein', '*Oc1ccc(-c2ccc(OC(=O)c3ccc(C4(c5ccc(C(*)=O)cc5)OC(=O)c5ccccc54)cc3)cc2)cc1', '{[<]C(=O)c3ccc(cc3)C4(OC(=O)c5ccccc54)c5ccc(cc5)C(=O)[<],[>]Oc1ccc(cc1)c2ccc(cc2)O[>]}', 580),
    ('Polyquinoline', '*c1ccc(-c2ccc(-c3cc(-c4ccccc4)c4cc(Oc5ccc6nc(*)cc(-c7ccccc7)c6c5)ccc4n3)cc2)cc1', '{[<]c1ccc(cc1)c2ccc(cc2)c3cc(c4ccccc4)c4cc(ccc4n3)Oc5ccc6nc(cc(c7ccccc7)c6c5)[>]}', 581),
    ('Polyphenolphthalein', '*Oc1ccc(C2(c3ccc(OC(=O)c4ccc(C5(c6ccc(C(*)=O)cc6)OC(=O)c6ccccc65)cc4)cc3)OC(=O)c3ccccc32)cc1', '{[<]C(=O)c4ccc(cc4)C5(OC(=O)c6ccccc65)c6ccc(cc6)C(=O)[<],[>]Oc1ccc(cc1)C2(OC(=O)c3ccccc32)c3ccc(cc3)O[>]}', 583),
    ('Polyimide', '*c1ccc(C(=O)c2ccc(N3C(=O)c4ccc(C(c5ccc6c(c5)C(=O)N(*)C6=O)(C(F)(F)F)C(F)(F)F)cc4C3=O)cc2)cc1', '{[<]c1ccc(cc1)C(=O)c2ccc(cc2)[<],[>]N3C(=O)c4ccc(cc4C3=O)C(C(F)(F)F)(C(F)(F)F)c5ccc6c(c5)C(=O)N(C6=O)[>]}', 584),
    ('Poly(quinoxaiine-2, 7- diylcarbonylquinoxaline- 7,2-diyl- l ,4- phenylene)', '*c1ccc(-c2cnc3ccc(C(=O)c4ccc5ncc(*)nc5c4)cc3n2)cc1', '{[<]c1ccc(cc1)c2cnc3ccc(cc3n2)C(=O)c4ccc5ncc(nc5c4)[>]}', 591),
    ('Polyphenolphthalein', '*Nc1ccc(Oc2ccc(NC(=O)c3ccc(C4(c5ccc(C(*)=O)cc5)OC(=O)c5ccccc54)cc3)cc2)cc1', '{[<]C(=O)c3ccc(cc3)C4(OC(=O)c5ccccc54)c5ccc(cc5)C(=O)[<],[>]Nc1ccc(cc1)Oc2ccc(cc2)N[>]}', 593),
    ('Polyquinoline', '*c1ccc(Oc2ccc(-c3cc(-c4ccc(Oc5ccc(-c6cc(*)nc7ccccc67)cc5)cc4)c4ccccc4n3)cc2)cc1', '{[<]c1ccc(cc1)Oc2ccc(cc2)[<],[>]c3cc(c4ccccc4n3)c4ccc(cc4)Oc5ccc(cc5)c6cc(nc7ccccc67)[>]}', 599),
    ('Poly(p-phenylene terephthalamide)', '*Nc1ccc(NC(=O)c2ccc(C(*)=O)cc2)cc1', '{[<]C(=O)c2ccc(cc2)C(=O)[<],[>]Nc1ccc(cc1)N[>]}', 600),
    ('Polyimide', '*c1ccc(N2C(=O)c3ccc(C(=O)c4ccc5c(c4)C(=O)N(*)C5=O)cc3C2=O)cc1', '{[<]c1ccc(cc1)N2C(=O)c3ccc(cc3C2=O)C(=O)c4ccc5c(c4)C(=O)N(C5=O)[>]}', 606),
    ('Poly(quinoxaline-2,7- diylsulfonylquinoxaline-7,2-diyl-1,4- phenylene)', '*c1ccc(-c2cnc3ccc(Oc4ccc5ncc(*)nc5c4)cc3n2)cc1', '{[<]c1ccc(cc1)c2cnc3ccc(cc3n2)S(=O)(=O)c4ccc5ncc(nc5c4)[>]}', 615),
    ('Polyetherimide', '*c1ccc(N2C(=O)c3ccc(Oc4ccc5c(c4)C(=O)N(*)C5=O)cc3C2=O)cc1', '{[<]c1ccc(cc1)[<],[>]N2C(=O)c3ccc(cc3C2=O)Oc4ccc5c(c4)C(=O)N(C5=O)[>]}', 615),
    ('Polyquinoline', '*c1ccc(-c2nc3ccc(Oc4ccc5nc(*)c(-c6ccccc6)c(-c6ccccc6)c5c4)cc3c(-c3ccccc3)c2-c2ccccc2)cc1', '{[<]c1ccc(cc1)Oc2ccc(cc2)[<],[>]c3cc(c4ccccc4n3)c4ccc(cc4)Oc5ccc(cc5)c6cc(nc7ccccc67)[>]}', 618),
    ('Polyquinoline', '*c1ccc(Oc2ccc(-c3nc4ccccc4c(-c4ccc(Oc5ccc(-c6c(-c7ccccc7)c(*)nc7ccccc67)cc5)cc4)c3-c3ccccc3)cc2)cc1', '{[<]c1ccc(cc1)[<],[>]c2nc3ccc(cc3c(c3ccccc3)c2c2ccccc2)Oc4ccc5nc(c(c6ccccc6)c(c6ccccc6)c5c4)[>]}', 618),
    ('Polyquinoline', '*c1ccc(-c2ccc(-c3nc4ccc(Oc5ccc6nc(*)c(-c7ccccc7)c(-c7ccccc7)c6c5)cc4c(-c4ccccc4)c3-c3ccccc3)cc2)cc1', '{[<]c1ccc(cc1)[<],[>]c2ccc(cc2)c3nc4ccc(cc4c(c4ccccc4)c3c3ccccc3)Oc5ccc6nc(c(c7ccccc7)c(c7ccccc7)c6c5)[>]}', 624),
    ('Poly(quinoxaline-2,7-diylquinoxaline-7,2-diyl-1,4-phenylene)', '*c1ccc(-c2cnc3ccc(-c4ccc5ncc(*)nc5c4)cc3n2)cc1', '{[<]c1ccc(cc1)c2cnc3ccc(cc3n2)c4ccc5ncc(nc5c4)[>]}', 645),
    ('Polyphenoiphthalein', '*Nc1ccc(C2(c3ccc(NC(=O)c4ccc(C5(c6ccc(C(*)=O)cc6)OC(=O)c6ccccc65)cc4)cc3)OC(=O)c3ccccc32)cc1', '{[<]C(=O)c4ccc(cc4)C5(OC(=O)c6ccccc65)c6ccc(cc6)C(=O)[<],[>]Nc1ccc(cc1)C2(OC(=O)c3ccccc32)c3ccc(cc3)N[>]}', 658),
    ("Poly[N,N'-(p,p'- oxydiphenylene)pyromellitimide]", '*Oc1ccc(-n2c(=O)c3cc4c(=O)n(-c5ccc(*)cc5)c(=O)c4cc3c2=O)cc1', '{[<]Oc1ccc(cc1)n2c(=O)c3cc4c(=O)n(c(=O)c4cc3c2=O)c5ccc(cc5)[>]}', 672),
    ('Polyphenolphthalein', '*C(=O)Nc1ccc(C2(c3ccc(NC(=O)c4ccc(C5(c6ccc(*)cc6)OC(=O)c6ccccc65)cc4)cc3)NC(=O)c3ccccc32)cc1', '{[<]NC(=O)Nc1ccc(cc1)C2(NC(=O)c3ccccc32)c3ccc(cc3)NC(=O)[>]}', 673),
    ("Poly(N,N'-(p,p'-carbonyldiphenylene) pyrornellitimide]", '*C(=O)c1ccc(-n2c(=O)c3cc4c(=O)n(-c5ccc(*)cc5)c(=O)c4cc3c2=O)cc1', '{[<]C(=O)c1ccc(cc1)n2c(=O)c3cc4c(=O)n(c(=O)c4cc3c2=O)c5ccc(cc5)[>]}', 685),
)


# ---------------------------------------------------------------------------
# Shorthand → bracket converter / 简写转换器
# ---------------------------------------------------------------------------

def shorthand_to_bracket(s: str) -> str:
    """Convert shorthand BigSMILES descriptors to bracket notation.

    Examples:
        {$}     → {[$]CC[$]}
        {<CCO>}    → {[<]CCO[>]}
    """
    result: list = []
    in_stoch = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == '{':
            in_stoch += 1
            result.append(ch)
            i += 1
        elif ch == '}':
            in_stoch -= 1
            result.append(ch)
            i += 1
        elif in_stoch > 0 and ch in ('$', '<', '>'):
            if i > 0 and s[i - 1] == '[':
                result.append(ch)
            else:
                result.append('[' + ch + ']')
            i += 1
        elif ch == '[':
            j = s.index(']', i)
            result.append(s[i:j + 1])
            i = j + 1
        else:
            result.append(ch)
            i += 1
    return ''.join(result)


# ---------------------------------------------------------------------------
# Public API / 公共 API
# ---------------------------------------------------------------------------

def load_dataset() -> List[Dict[str, Any]]:
    """加载数据集为字典列表。/ Load dataset as list of dicts."""
    return [
        {
            "name": name,
            "smiles": smiles,
            "bigsmiles": bigsmiles,
            "tg_k": tg_k,
        }
        for name, smiles, bigsmiles, tg_k in BICERANO_DATA
    ]


def get_names() -> List[str]:
    """获取所有聚合物名称。/ Get all polymer names."""
    return [entry[0] for entry in BICERANO_DATA]


def get_smiles() -> List[str]:
    """获取所有重复单元 SMILES。/ Get all repeat unit SMILES."""
    return [entry[1] for entry in BICERANO_DATA]


def get_bigsmiles() -> List[str]:
    """获取所有 BigSMILES 字符串。/ Get all BigSMILES strings."""
    return [entry[2] for entry in BICERANO_DATA]


def get_tg_values() -> List[int]:
    """获取所有 Tg 值 (K)。/ Get all Tg values in Kelvin."""
    return [entry[3] for entry in BICERANO_DATA]


def to_csv(path: str) -> None:
    """导出为 CSV 文件。/ Export to CSV file."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['name', 'smiles', 'bigsmiles', 'tg_k'])
        for entry in BICERANO_DATA:
            writer.writerow(entry)


def to_json(path: str) -> None:
    """导出为 JSON 文件。/ Export to JSON file."""
    data = load_dataset()
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def validate_all(verbose: bool = False) -> List[Dict[str, Any]]:
    """校验所有 BigSMILES 字符串。/ Validate all BigSMILES strings with checker.

    Returns list of failed entries (empty if all pass).
    """
    from src.bigsmiles.checker import check_bigsmiles

    failures = []
    for i, (name, smiles, bigsmiles, tg_k) in enumerate(BICERANO_DATA):
        ok = check_bigsmiles(bigsmiles, verbose=verbose)
        if not ok:
            failures.append({
                "index": i,
                "name": name,
                "bigsmiles": bigsmiles,
            })
    return failures


def summary() -> Dict[str, Any]:
    """数据集概况。/ Dataset summary statistics."""
    tg_values = get_tg_values()
    return {
        "total_entries": len(BICERANO_DATA),
        "tg_min_k": min(tg_values),
        "tg_max_k": max(tg_values),
        "tg_mean_k": round(sum(tg_values) / len(tg_values), 1),
        "unique_smiles": len(set(get_smiles())),
        "source": "Choi et al., Scientific Data 11, 363 (2024)",
    }


# ---------------------------------------------------------------------------
# CLI entry / 命令行入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    s = summary()
    print(f"Bicerano Tg Dataset")
    print(f"  Entries:     {s['total_entries']}")
    print(f"  Tg range:    {s['tg_min_k']}–{s['tg_max_k']} K")
    print(f"  Tg mean:     {s['tg_mean_k']} K")
    print(f"  Unique RU:   {s['unique_smiles']}")
    print(f"  Source:      {s['source']}")

    if "--validate" in sys.argv:
        print("\nValidating all BigSMILES...")
        fails = validate_all()
        if not fails:
            print(f"  All {s['total_entries']} entries PASS.")
        else:
            print(f"  {len(fails)} FAILURES:")
            for f2 in fails:
                print(f"    [{f2['index']}] {f2['name']}: {f2['bigsmiles']}")

    if "--csv" in sys.argv:
        to_csv("bicerano_tg.csv")
        print("\nExported to bicerano_tg.csv")

    if "--json" in sys.argv:
        to_json("bicerano_tg.json")
        print("\nExported to bicerano_tg.json")
