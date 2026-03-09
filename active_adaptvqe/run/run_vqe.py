"""VQE driver for running simulation with various controls."""

import sys

import itertools

from qiskit import QuantumCircuit

from utils import _to_symmer_QuantumState 

from custom_vqe.general.pauli_array import PauliArr	
from custom_vqe.circuit.circuit_ucc import _circ_UCCSD

from qiskit_nature.second_q.mappers import JordanWignerMapper

#
from qiskit.quantum_info import Statevector
from symmer import process, QuantumState, PauliwordOp
from symmer.operators.utils import (
    symplectic_to_string, safe_PauliwordOp_to_dict, safe_QuantumState_to_dict
)
from symmer.evolution import PauliwordOp_to_QuantumCircuit, get_CNOT_connectivity_graph, topology_match_score
from networkx.algorithms.cycles import cycle_basis
from scipy.optimize import minimize
from scipy.sparse import csc_array
from copy import deepcopy
import numpy as np
from typing import *
#

class run_vqe:
	r"""Create run_vqe object.

    This routine calculates either expectation value directly or uses sampling
    from the bitstring output.

    In sampling method, qiskit aersimulator is used. (Statevector or density-matrix)

	For direct evaluation of expectation values of observables, methods from `symmer` 
    package is used.
    Three such methods from `symmer` has been used here --
        (adapted from `symmer`).
        - symbolic direct: uses `symmer` to compute <psi|H|psi> directly using the QuantumState class. 
        - sparse_array: direct calculation by converting observable/state to sparse array.
        - dense_array: direct calculation by converting observable/state to dense array.

    """
    _default_strtype = "little-endian"

	def __init__(self, 
        expect_ops: PauliArr,	
        excitation_ops: np.ndarray[int]| FermionicOp | PauliArr,	
        ref_state: str,  
        if_csvqe: bool = False,
        measure_gr_type: str = 'qwc',
        expect_direct: bool = True ,
        direct_type: str = "sparse_array",
        sim_type: str = "statevector"

	) -> None:
		"""Initialize the variables for the run_vqe class.

		
		:param expect_ops: A PauliArr object denoting the observable. 
			:math:`\\sum_{i=1}^N C_i P_i`.
		:param excitation_ops: A numpy array of integers denoting the occupied and virtual orbitals
                            or Qiskit `FermionicOp` or pauli operators in the form of `PauliArr`.
		:param ref_state: A string denoting reference state as a bitstring of `0' and `1'. 
        :param csvqe: If contextual subspace to implement or not.
        :param measure_gr_type: Grouping strategy type for simultaneous measurement.
        :param expect_direct: True / False based on if evaluation type is direct expectation value or from sampling.
        :param direct_type: A string denoting type of method ('symbolic' or 'sparse_array' or 'dense_array') for direct
                        expectation value calculation.
		:param sim_type: If statevector or density-matrix type simulation to be used.

		"""
		if not expect_ops._if_hermitian: 
			raise TypeError("Operator is not Hermitian")

		assert sim_type in {"statevector", "density-matrix"}

        # array of observables in `PauliArr' form.
		self._measure_paulis = expect_ops
        # excitation operators on top of the ref. state.
        self._op_pool = excitation_ops
        # reference state.
		self._ref_state = ref_state
        # number of qubits.
        self._nqbits = len(ref_state)
        # Identity operator of the length of number of qubits.
        self._id_op = list("I" * self._nqbits)
        # grouping strategy for simultaneous measurement.
        self._measure_gr_type = measure_gr_type
        self._if_expectation = expect_direct
		self._sim_type = sim_type

	def _construct_ansatz(self) -> "QuantumCircuit" | "QuantumState":
		"""Compute ansatz as QuantumCircuit or QuantumState.

        Construct the ansatz as qiskit `QuantumCircuit' or `QuantumState' from `symmer'.

		:return: qiskit `QuantumCircuit' or symmer `QuantumState'.

		"""
        if self._if_expectation:
            qc = _circ_UCCSD(self._op_pool, self._ref_state) 
            Statevector()
            # symmer QuantumState
            symmer_qs = QuantumState.from_array()

        else:
            qc = QuantumCircuit(len(self._ref_state))

            if self._default_strtype  == "little-endian":
                id_occ = [bit for i, bit in enumerate(self._ref_state[::-1]) if bit == '1']
				qc.x(id_occ)
                qc += _circ_UCCSD(self._op_pool)

            elif self._default_strtype  == "big-endian":
                id_occ = [bit for i, bit in enumerate(self._ref_state) if bit == '1']
                qc.x(id_occ)
                qc += _circ_UCCSD(self._op_pool)

                return qc


#	def _calc_expectation_statevector(self, 
#		):
#
#		# dense numpy array.
#		(psi.conjugate()@ expect_op @ psi)
#
#		# sparse numpy array.
#
#
#	def _calc_partial_drv() -> :
#
#

    def _calc_E_grad(self, analytic_grad: bool = True) -> :
        """Compute energy gradient wrt a given parameter.


        """


    def _process_adapt_data() -> :
        """Process data from converged / last iterated stage.


        """


#
#


    @staticmethod
    def build_callback(callback_dict: dict):
        """Compute Callback function for tracking progress of the `VQE` calculation.





        """
        def my_callback(cur_paramvec):
            
            if isinstance(
            cur_cost = 

            cur_cost = estimator.run([(ansatz, hamiltonian, [current_vector])]).result()[0].data.evs[0]


        def my_callback(current_vector):
            cur_cost = estimator.run([(ansatz, hamiltonian, [current_vector])]).result()[0].data.evs[0]

            callback_dict["prev vector"] = current_vector
            callback_dict["Iterations done"] += 1
            callback_dict["Cost history"].append(cur_cost)

            print(
                f"Iterations done: {callback_dict['Iterations done']}, current cost: {cur_cost}",
                end="\r",
                flush=True,
            )


    @staticmethod
    def cost_func(
        params: list | np.ndarray,
        ansatz: QuantumCircuit,
        hamiltonian: SparsePauliOp,
        estimator: Estimator,
    ) -> float:
        """Compute cost function for `VQE` calculation."""
        # length of params must be equal to number of parameters in the circuit.
        assert ansatz.num_parameters == len(params)




	def _run(self, 
		sim_type: str = "statevector",
		measure_type: str = "sampling",

	) -> float:

		"""


		"""

        assert sim_type in {"statevector", "density-matrix"}, "Only statevector and density-matrix simulations supported"

        assert measure_type in {"expectation", "sampling"}, "Measurement type not supported"
























class run_adapt_vqe(run_vqe):
	r"""Create run_adapt_vqe object."""

	def __init__(self, 
        expect_ops: PauliArr,
        excitation_ops: np.ndarray[int]| FermionicOp | PauliArr,	
        ref_state: str,  
        measure_gr_type: str = 'qwc',
        expect_direct: bool = True,
        direct_type: str = "sparse_array",
        sim_type: str = "statevector"
        max_itr: int = 1e+04,
        adapt_grad_thr: float = 1e-06,
        adapt_type: str = "qeb-adapt"
        tetris_adapt_type: str = "qeb-adapt",
        n_qeb_cutoff: int = 4,
        verbose: bool = False
	) - > None:

		"""Initialize run_adapt_vqe_driver object.


		"""
		super.__init__(expect_ops, excitation_ops, ref_state, measure_gr_type, expect_direct, sim_type)


        self._max_itr = max_itr
        self._adapt_grad_thr = adapt_grad_thr

        assert adapt_type in {"fermion-adapt","qubit-adapt", "tetris-adapt", "qeb-adapt"}, "Adaptive simulation type not supported!"

        assert tetris_adapt_type in {"fermion-adapt", "qubit-adapt", "qeb-adapt"}, "Provided tetris-adapt type not supported!"

        self._adapt_type = adapt_type
        self._tetris_adapt_type = tetris_adapt_type
        self._n_qeb_cutoff = n_qeb_cutoff

        self._verbose = verbose

        # compatibility checks for adaptive simulation type and excitation operator type. 



    def _map_fermionic_adapt(self, to_Zchain: bool = True, exc_type: str = "uccsd") -> tuple[list]:
        """Compute operator mapping of operator pool for fermionic adapt.

        This calculates mapping of a fermionic excitation term (Op - h.c.)
        Currently, just supports UCC singles and doubles type excitations.
        As exact expressions for UCC singles and doubles are known, those are
        implemented as such without calling fermion-to-qubit mapper. First 
        order Trotterization is used thorughout.

        :param exc_type: A string denoting type of excitation operator.
        :param to_Zchain: True / False if `ZZZZ...' to be implemented as part of circuit.
                        For Fermionic case, it needs to be implemented, but if turned off, 
                        it generates QEB-adapt.
        :return: A tuple.
        """
        if exc_type not in ("ucc_singles", "ucc_doubles", "uccsd"):
            raise NotImplementedError("excitation other than ucc singles or doubles 
                not implemented yet.")

        # converting each fermionic excitation term to corresponding pauli term.
       
	    op_adapt = []
        coeff_adapt = []

        for i, operator in enumerate(self._op_pool): 
        
            if len(operator) == 2:
                op_1, op_2 = self._id_op, self._id_op
                id_X, id_Y = operator[0], operator[1]  
                if to_Zchain:
                    op_1[id_X + 1: id_Y ] = "Z" * (id_Y - id_X -1)                
                    op_2[id_X + 1: id_Y ] = "Z" * (id_Y - id_X -1)              

                op_1[id_X] = "X"
                op_1[id_Y] = "Y"
                op_2[id_X] = "Y"
                op_2[id_Y] = "X"

                # reversing for sticking to little-endian convention.
                op_1 = op_1[::-1]
                op_2 = op_2[::-1]
	
                op_append = [''.join(op_1), ''.join(op_2)]
                coeff_append = [0.5, -0.5]

                op_adapt.append(op_append)
                coeff_adapt.append(coeff_append)

            elif len(operator) == 4:
                # First four Pauli operators for UCC doubles conversion to pauli strings.
                op_1, op_2, op_3, op_4 = self._id_op, self._id_op, self._id_op, self._id_op  
                # H.c of First four Pauli operators for UCC doubles conversion to pauli strings.
                op_5, op_6, op_7, op_8 = self._id_op, self._id_op, self._id_op, self._id_op  

                op_append = [op_1, op_2, op_3, op_4, op_5, op_6, op_7, op_8]

                # location of i,j,k,l with i < j < k < l
                id_1, id_2, id_3, id_4 = operator[0], operator[1], operator[2], operator[3] 

                # Z-strings for UCC doubles.
                Z_p = "Z" * (id_2 - id_1 -1) 
                Z_r = "Z" * (id_4 - id_3 -1) 

                if to_Zchain: 
                    for op in op_append:
                        op[id_1+1: id_2] = Z_p
                        op[id_3+1: id_4] = Z_r

                # term 1 #
                op_1[id_1] = "X"
                op_1[id_2] = "Y"
                op_1[id_3] = "X"
                op_1[id_4] = "X"

                # term 2 #
                op_2[id_1] = "Y"
                op_2[id_2] = "X"
                op_2[id_3] = "X"
                op_2[id_4] = "X"

                # term 3 #
                op_3[id_1] = "Y"
                op_3[id_2] = "Y"
                op_3[id_3] = "Y"
                op_3[id_4] = "X"

                # term 4 #
                op_4[id_1] = "Y"
                op_4[id_2] = "Y"
                op_4[id_3] = "X"
                op_4[id_4] = "Y"

                # term 5 #
                op_5[id_1] = "X"
                op_5[id_2] = "X"
                op_5[id_3] = "Y"
                op_5[id_4] = "X"

                # term 6 #
                op_6[id_1] = "X"
                op_6[id_2] = "X"
                op_6[id_3] = "X"
                op_6[id_4] = "Y"

                # term 7 #
                op_7[id_1] = "Y"
                op_7[id_2] = "X"
                op_7[id_3] = "Y"
                op_7[id_4] = "Y"

                # term 8 #
                op_8[id_1] = "X"
                op_8[id_2] = "Y"
                op_8[id_3] = "Y"
                op_8[id_4] = "Y"

                # reversing for sticking to little-endian convention.
                op_1 = op_1[::-1]
                op_2 = op_2[::-1]
                op_3 = op_3[::-1]
                op_4 = op_4[::-1]
                op_5 = op_5[::-1]
                op_6 = op_6[::-1]
                op_7 = op_7[::-1]
                op_8 = op_8[::-1]

                op_append = [''.join(op_1), ''.join(op_2), ''.join(op_3), ''.join(op_4), ''.join(op_5), ''.join(op_6), ''.join(op_7), ''.join(op_8)] 

                coeff_append = [0.125, 0.125, 0.125, 0.125, -0.125, -0.125, -0.125, -0.125]
               
			    op_adapt.append(op_append)
                coeff_adapt.append(coeff_append)
        
        return (op_adapt, coeff_adapt)


    def _map_qubit_adapt(self) -> tuple[list]:
        """Compute operator mapping of operator pool for qubit adapt.

        :return: A tuple.
        """
        joint_fermion_ops = self._map_fermionic_adapt()

        ops_old = joint_fermion_ops[0]
        coeffs_old = joint_fermion_ops[1]

        ops_new = list(itertools.chain.from_iterable(ops_old)) 
        coeffs_new = list(itertools.chain.from_iterable(coeffs_old)) 

        return (ops_new, coeffs_new)

    def _map_qeb_adapt(self) -> tuple[list]:
        """Compute operator mapping of operator pool for qeb adapt.

        :return: A tuple.
        """
        return self._map_fermionic_adapt(to_Zchain = False)

    def _map_tetris_adapt(self) -> tuple[list]:
        """Compute operators for tetris-adapt vqe.

        :return: A tuple.
        """
        if self._tetris_adapt_type == "fermion-adapt":
            return self._map_fermionic_adapt()

        elif self._tetris_adapt_type == "qubit-adapt":
            return self._map_qubit_adapt()

        elif self._tetris_adapt_type == "qeb-adapt":
            return self._map_qeb_adapt()


    def _get_tetris_terms(self, gradarr: np.ndarray[float]) -> tuple[list]:
        """Compute operators for tetris-adapt at a given stage of simulation.

        :return: A tuple.
        """





	def _run_adapt(self) -> float:
		"""Run the adapt vqe simulations.

        :param analytic_grad: True / False if analytic gradient method to use or not.
        :return: 
		"""
		assert adapt_type in {"fermion-adapt","qubit-adapt", "tetris", "adapt"}

        cur_itr = 0 

        psi_cur = self._ref_state 

        while cur_itr < self._maxitr:
            qubit_oparr = self._op_pool

            # converting fermionic excitations to pauli operators.
            if self._adapt_type == "fermion-adapt":
                qubit_oparr = self._map_fermionic_adapt()

            # generating pauli strings for the fermionic excitations. 
            elif self._adapt_type == "qubit-adapt":
                qubit_oparr = self._map_qubit_adapt(self._op_pool)

            # converting fermionic excitations to qubit excitations. 
            elif self._adapt_type == "qeb-adapt":
                qubit_oparr = self._map_qeb_adapt(self._op_pool)

            # operators for tetris-adapt.
            elif self._adapt_type == "tetris-adapt":
                qubit_oparr = self._map_tetris_adapt()


            # array containing gradients of the operators in operator pool.
            op_gradarr = np.zeros(len(qubit_oparr))

            for i, operator in enumerate(qubit_oparr):
                op_grad = super()._calc_E_grad(operator, psi_cur)
                op_gradarr[i] = op_grad

            # checking convergence criteria meeting
			# TODO (recheck convergence criteria)
            if grad_norm(op_grad) < self._adapt_grad_thr:
                self._process_adapt_data()
                break

			# find the operator with the largest gradient.
            op_select = qubit_oparr[np.argmax(abs(op_gradarr))] 

            # operator selection for tetris-adapt vqe.
            oparr_qeb_adapt = self._get_tetris_terms(op_gradarr) 

            psi_next = 

            # run VQE with current set of parameters.
            self._run()





		


















