"""
Nucleic acid sequence -> SMILES converter.
Adapted from bigsmiles_project/sequence_to_bigsmiles.py

Public API:
    validate_sequence(seq, seq_type) -> str
    build_full_smiles(seq, seq_type, direction) -> str
    sequence_to_smiles(seq, seq_type, direction) -> dict
"""

import re
from typing import Dict

# ---------------------------------------------------------------------------
# Nucleotide SMILES fragment dictionaries
# ---------------------------------------------------------------------------

DNA_BASES = set("ATGC")
RNA_BASES = set("AUGC")

# DNA internal nucleotides (with 3',5'-phosphodiester linkage)
DNA_INTERNAL = {
    "A": "CC3OC(n1cnc2c(N)ncnc12)CC3OP(=O)(O)O",
    "T": "CC3OC(n1cc(C)c(=O)[nH]c1=O)CC3OP(=O)(O)O",
    "G": "CC3OC(n1cnc2c(=O)[nH]c(N)nc12)CC3OP(=O)(O)O",
    "C": "CC3OC(n1ccc(N)nc1=O)CC3OP(=O)(O)O",
}

# DNA 3'-terminal nucleotides (free 3'-OH, no trailing phosphate)
DNA_TERMINAL = {
    "A": "CC3OC(n1cnc2c(N)ncnc12)CC3O",
    "T": "CC3OC(n1cc(C)c(=O)[nH]c1=O)CC3O",
    "G": "CC3OC(n1cnc2c(=O)[nH]c(N)nc12)CC3O",
    "C": "CC3OC(n1ccc(N)nc1=O)CC3O",
}

# RNA internal nucleotides (2'-OH: CC -> C(O)C, T -> U)
RNA_INTERNAL = {
    "A": "CC3OC(n1cnc2c(N)ncnc12)C(O)C3OP(=O)(O)O",
    "U": "CC3OC(n1ccc(=O)[nH]c1=O)C(O)C3OP(=O)(O)O",
    "G": "CC3OC(n1cnc2c(=O)[nH]c(N)nc12)C(O)C3OP(=O)(O)O",
    "C": "CC3OC(n1ccc(N)nc1=O)C(O)C3OP(=O)(O)O",
}

RNA_TERMINAL = {
    "A": "CC3OC(n1cnc2c(N)ncnc12)C(O)C3O",
    "U": "CC3OC(n1ccc(=O)[nH]c1=O)C(O)C3O",
    "G": "CC3OC(n1cnc2c(=O)[nH]c(N)nc12)C(O)C3O",
    "C": "CC3OC(n1ccc(N)nc1=O)C(O)C3O",
}

# Single nucleotide monomer SMILES (for monomer-level prediction)
NUCLEOTIDE_MONOMERS = {
    "DNA": {
        "A": "c1nc(N)c2ncn(C3CC(O)C(CO)O3)c2n1",
        "T": "Cc1c[nH]c(=O)[nH]c1=O",
        "G": "c1nc2c(nc(N)nc2[nH]1)O",
        "C": "c1cc(N)nc(=O)[nH]1",
    },
    "RNA": {
        "A": "c1nc(N)c2ncn(C3OC(CO)C(O)C3O)c2n1",
        "U": "c1c[nH]c(=O)[nH]c1=O",
        "G": "c1nc2c(nc(N)nc2[nH]1)O",
        "C": "c1cc(N)nc(=O)[nH]1",
    },
}


def validate_sequence(seq: str, seq_type: str = "DNA") -> str:
    """Validate and clean a nucleic acid sequence.

    Args:
        seq: Input sequence string.
        seq_type: "DNA" or "RNA".

    Returns:
        Cleaned uppercase sequence.

    Raises:
        ValueError: Invalid base character found.
    """
    seq = seq.strip().upper()
    seq = re.sub(r"^[35]'[- ]?", "", seq)
    seq = re.sub(r"[- ]?[35]'$", "", seq)

    valid_bases = DNA_BASES if seq_type == "DNA" else RNA_BASES
    for i, base in enumerate(seq):
        if base not in valid_bases:
            raise ValueError(
                f"Invalid base '{base}' at position {i + 1} "
                f"for {seq_type}. Valid: {', '.join(sorted(valid_bases))}"
            )
    if len(seq) == 0:
        raise ValueError("Empty sequence")
    return seq


def build_full_smiles(seq: str, seq_type: str = "DNA", direction: str = "5to3") -> str:
    """Build complete atom-level SMILES by nucleotide concatenation.

    Logic: 5'-OH + internal[base] * (N-1) + terminal[base] * 1

    Args:
        seq: Validated uppercase sequence.
        seq_type: "DNA" or "RNA".
        direction: "5to3" or "3to5".

    Returns:
        Complete SMILES string.
    """
    if direction == "3to5":
        seq = seq[::-1]

    if seq_type == "DNA":
        internal, terminal = DNA_INTERNAL, DNA_TERMINAL
    else:
        internal, terminal = RNA_INTERNAL, RNA_TERMINAL

    full_smiles = "O"  # 5'-OH
    for i, base in enumerate(seq):
        if i < len(seq) - 1:
            full_smiles += internal[base]
        else:
            full_smiles += terminal[base]

    return full_smiles


def get_monomer_smiles(base: str, seq_type: str = "DNA") -> str:
    """Get monomer-level SMILES for a single nucleotide base.

    Args:
        base: Single character (A/T/G/C for DNA, A/U/G/C for RNA).
        seq_type: "DNA" or "RNA".

    Returns:
        Monomer SMILES string.
    """
    base = base.upper()
    monomers = NUCLEOTIDE_MONOMERS.get(seq_type, NUCLEOTIDE_MONOMERS["DNA"])
    if base not in monomers:
        raise ValueError(f"Unknown base '{base}' for {seq_type}")
    return monomers[base]


def sequence_to_smiles(
    seq: str,
    seq_type: str = "DNA",
    direction: str = "5to3",
) -> Dict:
    """Convert nucleic acid sequence to SMILES representations.

    Returns both the full concatenated SMILES and per-base monomer SMILES.

    Args:
        seq: Nucleic acid sequence (e.g., "ACGT").
        seq_type: "DNA" or "RNA".
        direction: "5to3" or "3to5".

    Returns:
        Dict with keys: sequence, seq_type, length, full_smiles, monomers.
    """
    clean = validate_sequence(seq, seq_type)
    full = build_full_smiles(clean, seq_type, direction)

    monomers = []
    for base in clean:
        monomers.append({
            "base": base,
            "smiles": get_monomer_smiles(base, seq_type),
        })

    return {
        "sequence": clean,
        "seq_type": seq_type,
        "length": len(clean),
        "full_smiles": full,
        "monomers": monomers,
    }
