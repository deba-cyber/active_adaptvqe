"""Circuit construction for UCCSD ansatz."""

import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister, Parameter
from circuit import _circ_exp_pauli 


def _circ_UCCSD(
    qc: QuantumCircuit,
    exc_trms: np.ndarray,
    exc_coeffs: np.ndarray[Parameter],
    ctrl_Ry: bool = False,
    fswap: bool = False
) -> None:
    """Construct UCCSD circuit as qiskit QuantumCircuit.

    Updates the `QuantumCircuit' in-place.

    :param qc: A qiskit QuantumCircuit. 
    :exc_trms: A numpy array of pauli strings corresponding to excitation operator.
    :exc_coeffs: A numpy array of floats corresponding to phase factors, imaginary `i' is not 
                considered.
    :param ctrl_Ry: True / False if circuit to be constructed based on controlled Ry
                    method.
    :param fswap: True / False if Fermionic Swap is to be invoked. 

    """

    for i, exc in enumerate(exc_trms):
        if fswap and ctrl_Ry: 
            raise NotImplementedError("Fermionic Swap and controlled-Ry circuit 
                not implemented yet.")

        if fswap and not ctrl_Ry:
            raise NotImplementedError("Fermionic Swap and controlled-Ry circuit 
                not implemented yet.")

        if ctrl_Ry and not fswap:
            raise NotImplementedError("Fermionic Swap and controlled-Ry circuit 
                not implemented yet.")

        if not fswap and not ctrl_Ry:
            _circ_exp_pauli(qc, exc, exc_coeffs[i])




