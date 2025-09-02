# A code to solve the toric code via exact diagonalization
# Serves to illustrate some of the phenomenology of the model and
# to provide a benchmark for more advanced methods.

"""
The Hamiltonian is given by:
H= - sum_s A_s - sum_p B_p
where
A_s = product of sigma_x around vertex s (star operator)
B_p = product of sigma_z around plaquette p (plaquette operator)
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import argparse

I = sp.csr_matrix(([1.0, 1.0], ([0, 1], [0, 1])), shape=(2, 2))
X = sp.csr_matrix(([1.0, 1.0], ([0, 1], [1, 0])), shape=(2, 2))
Z = sp.csr_matrix(([1.0, -1.0], ([0, 1], [0, 1])), shape=(2, 2))




def get_edge_indices(Lx, Ly):
    # Initialize a Lx x Ly lattice. 
    """
    Output list of qubits and their positions.
    Essentially mapping edges to qubits.
    Here's a diagram of an example lattice 
    y
     +---+---+---+
      | 6 | 7 | 8 |
     +---+---+---+
      | 3 | 4 | 5 |
     +---+---+---+
      | 0 | 1 | 2 |
     +---+---+---+--> x



    """
    def h(x, y): # horizontal edges
        return (y % Ly) * Lx + (x % Lx)
    def v(x, y): # vertical edges are after horizontal edges in list
        return Lx * Ly + (y % Ly) * Lx + (x % Lx)

    return h, v
    

def get_star_edges(Lx, Ly, x, y):
    h, v = get_edge_indices(Lx, Ly)
    return [h(x, y), h(x-1, y), v(x, y), v(x, y-1)]


def get_plaquette_edges(Lx, Ly, x, y):
    ## Get the four edges around a plaquette whose lower-left corner is at (x,y)
    h, v = get_edge_indices(Lx, Ly)
    return [h(x, y), v(x+1, y), h(x, y+1), v(x, y)]



def string_operator(Nedges_tot, edges : list, op_type : str ='X' ) -> np.ndarray:
    """
    Construct a string operator (product of X or Z) on given edges.
    edges: list of edge indices
    op_type: 'X' or 'Z'
    """
    #print(f"Constructing {op_type}-string on edges {edges} for {Nedges_tot} qubits")

    op_list = []
    for i in range(Nedges_tot):
        if i in edges:
            if op_type == 'X':
                op_list.append(X)
            elif op_type == 'Z':
                op_list.append(Z)
            else:
                raise ValueError("op_type must be 'X' or 'Z'")
        else:
            op_list.append(I)

    # Kronecker product to build the full operator
    #print("Building full operator via Kronecker products...")
#    print("op_list = ", op_list)
    full_op = op_list[0]
    for mat in op_list[1:]:
        full_op = sp.kron(full_op, mat, format='csr')
        #print("Full operator shape: ", full_op.shape)


        
    return full_op


## helper functions for strings of X and Z
def x_string(Nedges_tot, sites):
    return string_operator(Nedges_tot, sites, op_type='X')

def z_string(Nedges_tot : int, sites : list) -> np.ndarray:
    """ Construct Z string operator on given sites.
    sites: list of qubit indices (0-based)
    
    """
    return string_operator(Nedges_tot, sites, op_type='Z')

## string ops for star and plaquette

def star_operator(Lx, Ly, x, y):
    """Star operator A_s at vertex (x,y) - product of X on four edges touching vertex."""
    edges = get_star_edges(Lx, Ly, x, y)
    return x_string(2*Lx*Ly, edges)

def plaquette_operator(Lx, Ly, x, y):
    """Plaquette operator B_p at plaquette with lower-left corner (x,y) - product of Z on four edges around plaquette."""
    edges = get_plaquette_edges(Lx, Ly, x, y)
    return z_string(2*Lx*Ly, edges)



### functions to create op strings for non-contractible loops on torus

def z_loop_x(Lx, Ly, y0):
    """Non-contractible Z loop along x (row y0) on horizontal edges."""
    h, _ = get_edge_indices(Lx, Ly)
    return [h(x, y0) for x in range(Lx)]

def z_loop_y(Lx, Ly, x0):
    """Non-contractible Z loop along y (column x0) on vertical edges."""
    _, v = get_edge_indices(Lx, Ly)
    return [v(x0, y) for y in range(Ly)]

def x_loop_x(Lx, Ly, y0):
    # dual loop around x: vertical edges across x at fixed row y0
    _, v = get_edge_indices(Lx, Ly)
    return [v(x, y0) for x in range(Lx)]

def x_loop_y(Lx, Ly, x0):
    # dual loop around y: horizontal edges across y at fixed column x0
    h, _ = get_edge_indices(Lx, Ly)
    return [h(x0, y) for y in range(Ly)]


def build_hamiltonian(Lx, Ly, Je=1.0, Jm=1.0, verbose=False): 
    """
    H = -Je * sum_s A_s  - Jm * sum_p B_p
    where each A_s is an X-string over the 4 star edges,
          each B_p is a Z-string over the 4 plaquette edges.
    """
    N = 2 * Lx * Ly  # total number of qubits (edges)
    dim = 1 << N     # dimension of Hilbert space

    print(f"Building Hamiltonian for {Lx}x{Ly} lattice with {N} qubits (dim={dim})")


    H = sp.csr_matrix((dim, dim), dtype=np.float64)
    # Star terms
    for x in range(Lx):
        for y in range(Ly):
            A_s = star_operator(Lx, Ly, x, y)
            print(A_s.shape)

            H -= Je * A_s
            if verbose:
                print(f"Added star operator at ({x},{y})")

    # Plaquette terms
    for x in range(Lx):
        for y in range(Ly):
            B_p = plaquette_operator(Lx, Ly, x, y)
            H -= Jm * B_p
            if verbose:
                print(f"Added plaquette operator at ({x},{y})")

    return H

def expval(op, state):
    """ Expectation value <state|op|state> """
    return np.vdot(state, op @ state).real






        
def plot_lattice(Lx, Ly):
    """ Plot the Lx x Ly lattice with edges labeled by qubit indices. """
    h, v = get_edge_indices(Lx, Ly)

    #plt.figure(figsize=(Lx, Ly))
    for y in range(Ly):
        for x in range(Lx):
            # Draw horizontal edges
            plt.plot([x, x+1], [y, y], 'k-')
            plt.text(x + 0.5, y + 0.1, str(h(x, y)), ha='center', va='bottom', color='blue')
            # Draw vertical edges
            plt.plot([x, x], [y, y+1], 'k-')
            plt.text(x - 0.1, y + 0.5, str(v(x, y)), ha='right', va='center', color='red')

    plt.xlim(-0.5, Lx + 0.5)
    plt.ylim(-0.5, Ly + 0.5)
#    plt.gca().set_aspect('equal')
    plt.title(f'Toric Code Lattice {Lx}x{Ly}')
    plt.axis('off')
    plt.show()


def main():

    p = argparse.ArgumentParser(description="Toric code ED (no bit ops; sparse Kronecker only).")
    p.add_argument("--Lx", type=int, default=2)
    p.add_argument("--Ly", type=int, default=2)
    p.add_argument("--Je", type=float, default=1.0)
    p.add_argument("--Jm", type=float, default=1.0)
    p.add_argument("--k", "--num_eigs", type=int, default=4, help="# of lowest eigenpairs")
    p.add_argument("--tol", type=float, default=1e-8, help="degeneracy tolerance")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    
    Lx, Ly = args.Lx, args.Ly
    Je, Jm = args.Je, args.Jm

    # Plot the lattice
    N=2*Lx*Ly
    k=args.k
    dim = 1 << N # dimension of Hilbert space
    
    # Build Hamiltonian
    H = build_hamiltonian(Lx, Ly, Je=Je, Jm=Jm, verbose=False)
    print(f"Hamiltonian dimension: {H.shape}")

    # Diagonalize Hamiltonian
    k = min(args.k, dim - 1)
    print(f"Diagonalizing… ({k} lowest eigenpairs)")


    Zx = z_string(N, z_loop_x(Lx, Ly, y0=0))
    Zy = z_string(N, z_loop_y(Lx, Ly, x0=0))
    Xx = x_string(N, x_loop_x(Lx, Ly, y0=0))
    Xy = x_string(N, x_loop_y(Lx, Ly, x0=0))
    epsx = 1e-8;epsy=2e-8
    H_tiebreak = H - epsx*Zy - epsy*Zx   # both commute with H

    evals, evecs = spla.eigsh(H_tiebreak, k=k, which="SA")
    order = np.argsort(evals)
    evals, evecs = evals[order], evecs[:, order]

    #evals, evecs = np.linalg.eigh(H.toarray())
    print("Lowest 10 eigenvalues:")
    print(evals[:10])

    # Check expectation values of star and plaquette operators in ground state
    ground_state = evecs[:, 0]
    for x in range(Lx):
        for y in range(Ly):
            A_s = star_operator(Lx, Ly, x, y)
            B_p = plaquette_operator(Lx, Ly, x, y)
            exp_A = expval(A_s, ground_state)
            exp_B = expval(B_p, ground_state)
            print(f"<A_s({x},{y})> = {exp_A:.4f}, <B_p({x},{y})> = {exp_B:.4f}")

    # Check expectation values of non-contractible loops in ground state

    # for x0 in range(Lx):
    #     Zx = z_loop_x(Lx, Ly, x0)
    #     Xx = x_loop_x(Lx, Ly, x0)
    #     exp_Zx = expval(z_string(2*Lx*Ly, Zx), ground_state)
    #     exp_Xx = expval(x_string(2*Lx*Ly, Xx), ground_state)
    #     print(f"<Z loop x={x0}> = {exp_Zx:.4f}, <X loop x={x0}> = {exp_Xx:.4f}")

    # for y0 in range(Ly):
    #     Zy = z_loop_y(Lx, Ly, y0)
    #     Xy = x_loop_y(Lx, Ly, y0)
    #     exp_Zy = expval(z_string(2*Lx*Ly, Zy), ground_state)
    #     exp_Xy = expval(x_string(2*Lx*Ly, Xy), ground_state)
    #     print(f"<Z loop y={y0}> = {exp_Zy:.4f}, <X loop y={y0}> = {exp_Xy:.4f}")
        
    # The ground state should have <A_s> = 1 and <B_p> = 1 for all stars and plaquettes

    Zx = z_string(N, z_loop_x(Lx, Ly, y0=0))
    Zy = z_string(N, z_loop_y(Lx, Ly, x0=0))
    Xx = x_string(N, x_loop_x(Lx, Ly, y0=0))
    Xy = x_string(N, x_loop_y(Lx, Ly, x0=0))

    print("\nTopological loop expectations on lowest states:")
    for i in range(min(k, 8)):
        v = evecs[:, i]
        print(
            "  state {:d}: <Zx>={:+.6f}  <Zy>={:+.6f}  <Xx>={:+.6f}  <Xy>={:+.6f}".format(
                i, expval(Zx, v), expval(Zy, v), expval(Xx, v), expval(Xy, v)
            )
        )

    # Stabilizer checks on |E0>
    v0 = evecs[:, 0]
    As_vals, Bp_vals = [], []
    for y in range(Ly):
        for x in range(Lx):
            As_vals.append(expval(x_string(N, get_star_edges(Lx, Ly, x, y)), v0))
            Bp_vals.append(expval(z_string(N, get_plaquette_edges(Lx, Ly, x, y)), v0))
    print("\nGround-state stabilizers (should be ~+1):")
    print(f"  <A_s>: mean={np.mean(As_vals):+.6f}, min={np.min(As_vals):+.6f}, max={np.max(As_vals):+.6f}")
    print(f"  <B_p>: mean={np.mean(Bp_vals):+.6f}, min={np.min(Bp_vals):+.6f}, max={np.max(Bp_vals):+.6f}")

    
    plot_lattice(Lx, Ly)
if __name__ == "__main__":
    main()
    
