#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ---------- single-/two-qubit gates ----------

HAD = (1/np.sqrt(2)) * np.array([[1, 1],[1,-1]], dtype=np.complex128)
Xg  = np.array([[0,1],[1,0]], dtype=np.complex128)
Zg  = np.array([[1,0],[0,-1]], dtype=np.complex128)
CNOT = np.array([[1,0,0,0],
                 [0,1,0,0],
                 [0,0,0,1],
                 [0,0,1,0]], dtype=np.complex128)

# lattice: edge indexing and geometry 
def edge_indices(Lx, Ly):
    def h(x, y):  # horizontal edge from (x,y)->(x+1,y)
        return (y % Ly) * Lx + (x % Lx)
    def v(x, y):  # vertical edge from (x,y)->(x,y+1)
        return Lx * Ly + (y % Ly) * Lx + (x % Lx)
    return h, v

def star_edges(Lx, Ly, x, y):
    # Get the four edges touching vertex (x,y)
    h, v = edge_indices(Lx, Ly)
    return [h(x, y), h(x-1, y), v(x, y), v(x, y-1)]

def plaquette_edges(Lx, Ly, x, y):
    ## Get the four edges around a plaquette whose lower-left corner is at (x,y)
    h, v = edge_indices(Lx, Ly)
    return [h(x, y), v(x+1, y), h(x, y+1), v(x, y)]

def edge_segment_coords(e, Lx, Ly):
    """Return endpoints ((x0,y0),(x1,y1)) for edge index e."""
    H = Lx * Ly
    if e < H:
        # horizontal
        x = e % Lx
        y = e // Lx
        
        return (x, y), (x+1, y)
    else:
        # vertical (remember the vertical edges
        # are indexed only after the horizontal edges)
        ei = e - H
        x = ei % Lx
        y = ei // Lx
        return (x, y), (x, y+1)

# ---------- apply gates to state vector (no bit ops) ----------
def apply_1q(psi, U, q, N):
    psi_perm = np.moveaxis(psi.reshape((2,)*N), q, -1)
    psi_perm = np.einsum('ab,...b->...a', U, psi_perm)
    psi_out  = np.moveaxis(psi_perm, -1, q).reshape(-1)
    return psi_out

def apply_2q(psi, U4, q1, q2, N):
    if q1 == q2: raise ValueError("q1 and q2 must differ")
    psi_t = psi.reshape((2,)*N)
    # move q1→axis -2, q2→axis -1
    psi_perm = np.moveaxis(psi_t, (q1, q2), (-2, -1))
    psi_blocks = psi_perm.reshape(-1, 4) @ U4.T
    psi_perm = psi_blocks.reshape(psi_perm.shape)
    psi_t = np.moveaxis(psi_perm, (-2, -1), (q1, q2))
    return psi_t.reshape(-1)

#  projector preparation on edges 
def A_s_apply(psi, edges, N):
    phi = psi
    for q in edges:
        phi = apply_1q(phi, Xg, q, N)
    return phi

def prepare_ground_state(Lx, Ly):
    """Start with |0..0>, then apply (I + A_s) for all stars (commuting projectors)."""
    N = 2*Lx*Ly
    psi = np.zeros(1<<N, dtype=np.complex128)
    psi[0] = 1.0
    for y in range(Ly):
        for x in range(Lx):
            psi = psi + A_s_apply(psi, star_edges(Lx, Ly, x, y), N)
            psi = psi / np.linalg.norm(psi)
    #print(f"Prepared ground state |GS> = {psi} with norm {np.linalg.norm(psi):.6f}")
    return psi

#  expectations values 
def expval_pauli_string_edges(psi, edges, which, N):
    U = Xg if which.upper() == "X" else Zg
    phi = psi
    for q in edges:
        phi = apply_1q(phi, U, q, N)
    return float(np.vdot(psi, phi).real)

def compute_As_Bp(psi, Lx, Ly):
    N = 2*Lx*Ly
    As = np.zeros((Ly, Lx)); Bp = np.zeros((Ly, Lx))
    for y in range(Ly):
        for x in range(Lx):
            As[y, x] = expval_pauli_string_edges(psi, star_edges(Lx,Ly,x,y), "X", N)
            Bp[y, x] = expval_pauli_string_edges(psi, plaquette_edges(Lx,Ly,x,y), "Z", N)
    return As, Bp

def z_loop_x_edges(Lx, Ly, y0):
    h,_ = edge_indices(Lx, Ly);  return [h(x, y0) for x in range(Lx)]
def z_loop_y_edges(Lx, Ly, x0):
    _,v = edge_indices(Lx, Ly);  return [v(x0, y) for y in range(Ly)]
def x_loop_x_edges(Lx, Ly, y0):
    _,v = edge_indices(Lx, Ly);  return [v(x, y0) for x in range(Lx)]
def x_loop_y_edges(Lx, Ly, x0):
    h,_ = edge_indices(Lx, Ly);  return [h(x0, y) for y in range(Ly)]

def apply_Z_path_edges(psi, edges, N):
    out = psi
    for q in edges:
        out = apply_1q(out, Zg, q, N)
    return out

def apply_X_path_edges(psi, edges, N):
    out = psi
    for q in edges:
        out = apply_1q(out, Xg, q, N)
    return out

# ---------- anyon helpers ----------
def e_anyons_from_As(As, tol=1e-8):
    """Return list of vertex coords with A_s ≈ -1 (e anyons)."""
    ys, xs = np.where(As < -1 + tol)
    return list(zip(xs.tolist(), ys.tolist()))

def m_anyons_from_Bp(Bp, tol=1e-8):
    """Return list of plaquette coords with B_p ≈ -1 (m anyons)."""
    ys, xs = np.where(Bp < -1 + tol)
    return list(zip(xs.tolist(), ys.tolist()))

# ---------- plotting ----------
def draw_lattice(Lx, Ly, label_edges=False,
                 highlight_Z=None, highlight_X=None,
                 e_anyons=None, m_anyons=None,
                 title=None):
    highlight_Z = set(highlight_Z or [])
    highlight_X = set(highlight_X or [])
    e_anyons = e_anyons or []
    m_anyons = m_anyons or []

    h, v = edge_indices(Lx, Ly)

    plt.figure(figsize=(1.2*Lx+2, 1.2*Ly+2))
    ax = plt.gca()

    # draw edges
    for y in range(Ly):
        for x in range(Lx):
            # horizontal
            e = h(x,y)
            (x0,y0),(x1,y1) = edge_segment_coords(e, Lx, Ly)
            lw = 4 if e in highlight_X or e in highlight_Z else 1.5
            color = "#2c3e50"
            if e in highlight_Z and e in highlight_X:
                color = "#8e44ad"  # both
            elif e in highlight_Z:
                color = "#27ae60"  # green Z
            elif e in highlight_X:
                color = "#e67e22"  # orange X
            ax.plot([x0,x1],[y0,y1], color=color, lw=lw)
            if label_edges:
                #print(f"Edge {e}: ({x0},{y0}) to ({x1},{y1})")
                ax.text((x0+x1)/2, y0-0.12, str(e), color="#2c3e50",
                        ha="center", va="top", fontsize=8)

            # vertical
            e = v(x,y)
            (x0,y0),(x1,y1) = edge_segment_coords(e, Lx, Ly)
            lw = 4 if e in highlight_X or e in highlight_Z else 1.5
            color = "#2c3e50"
            if e in highlight_Z and e in highlight_X:
                color = "#8e44ad"
            elif e in highlight_Z:
                color = "#27ae60"
            elif e in highlight_X:
                color = "#e67e22"
            ax.plot([x0,x1],[y0,y1], color=color, lw=lw)
            if label_edges:
                #print(f"Edge {e}: ({x0},{y0}) to ({x1},{y1})")
                ax.text(x0-0.12, (y0+y1)/2, str(e), color="#2c3e50",
                        ha="right", va="center", fontsize=8)

    # draw vertices
    for y in range(Ly+1):
        for x in range(Lx+1):
            ax.plot(x, y, 'o', color="#7f8c8d", ms=4)

    # draw e anyons (red dots at vertices)
    for (x,y) in e_anyons:
        ax.plot(x, y, 'o', color="#c0392b", ms=10, mew=1)

    # draw m anyons (blue squares at plaquette centers)
    for (x,y) in m_anyons:
        ax.plot(x+0.5, y+0.5, 's', color="#2980b9", ms=10, mew=1)

    ax.set_aspect('equal'); ax.set_xlim(-0.6, Lx+0.6); ax.set_ylim(-0.6, Ly+0.6)
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title)
    plt.tight_layout()
    plt.show()

# ---------- demos ----------
def show_noncontractible_loops(Lx, Ly,label_edges):
    N = 2*Lx*Ly
    psi = prepare_ground_state(Lx, Ly)
    p = probs(psi)
    Zx = z_loop_x_edges(Lx, Ly, y0=0)
    Zy = z_loop_y_edges(Lx, Ly, x0=0)
    Xx = x_loop_x_edges(Lx, Ly, y0=0)
    Xy = x_loop_y_edges(Lx, Ly, x0=0)
    print(f"Zx={Zx}")
    print(f"Xx={Xx}")
    print("Non-contractible loop expectations on |GS|:")
    print(" ⟨Zx⟩={:+.6f} ⟨Zy⟩={:+.6f} ⟨Xx⟩={:+.6f} ⟨Xy⟩={:+.6f}".format(
        expval_pauli_string_edges(psi, Zx, "Z", N),
        expval_pauli_string_edges(psi, Zy, "Z", N),
        expval_pauli_string_edges(psi, Xx, "X", N),
        expval_pauli_string_edges(psi, Xy, "X", N)))


    draw_lattice(Lx, Ly, label_edges=label_edges,
                 highlight_Z=set(Zx)|set(Zy),
                 highlight_X=set(Xy)|set(Xx),
                 title="Non-contractible Z (green) & X (orange) loops\n (Note X loops on dual lattice)")
    
    # draw_lattice(Lx, Ly, label_edges=label_edges,
    #              highlight_Z=set(Zx)|set(Zy),
    #              highlight_X=set(Xx)|set(Xy),
    #              title="Non-contractible Z (green) & X (orange) loops")

def braid_demo(Lx, Ly,label_edges):
    """
    Create an m-pair with a single X on one edge.
    Define Z loop around one of those plaquettes (contractible B_p).
    Compare states: Z_loop (X_edge |GS>)  vs  X_edge (Z_loop |GS>).
    They should differ by a global -1 → inner product ≈ -1.
    """
    N = 2*Lx*Ly
    psi0 = prepare_ground_state(Lx, Ly)

    # choose an edge and the two adjacent plaquettes
    h, v = edge_indices(Lx, Ly)
    eX = h(0, 0)  
    # Adjacent plaquettes: (0,0) and (0, Ly-1) due to periodicity
    p0 = (0, 0)
    Zloop = plaquette_edges(Lx, Ly, *p0)

    # Build the two orderings
    psi_X_then_Z = apply_Z_path_edges(apply_X_path_edges(psi0, [eX], N), Zloop, N)
    psi_Z_then_X = apply_X_path_edges(apply_Z_path_edges(psi0, Zloop, N), [eX], N)
    ov = np.vdot(psi_X_then_Z, psi_Z_then_X)
    print(f"Braiding phase ⟨ XZ | ZX ⟩ = {ov:.6f}  (expected ~ -1)")

    # Visuals: (i) after X (two m anyons), (ii) Z loop we braid with.
    psi_after_X = apply_X_path_edges(psi0, [eX], N)
    As, Bp = compute_As_Bp(psi_after_X, Lx, Ly)
    e_list = e_anyons_from_As(As); m_list = m_anyons_from_Bp(Bp)
    draw_lattice(Lx, Ly,
                 highlight_X=[eX], highlight_Z=Zloop,
                 e_anyons=e_list, m_anyons=m_list,label_edges=label_edges,
                 title="m-pair from single X edge (orange); Z loop (green) around one m")

def probs(psi):
    """Return probabilities |psi|^2."""
    return np.abs(psi)**2

def pretty_print_probs(p):
    for i, pi in enumerate(p):

        if pi > 1e-6:
            print(f" state = {i}  |{i:0{len(p).bit_length()-1}b}>: {pi:.16f}")
def prob_demo(Lx,Ly):
    N = 2*Lx*Ly
    psi0 = prepare_ground_state(Lx, Ly)
    p0 = probs(psi0)
    print("Probabilities in |GS> (nonzero only):")
    pretty_print_probs(p0)

    # Apply single X on edge eX
    h, v = edge_indices(Lx, Ly)
    eX = h(0, 0)
    psi_X = apply_X_path_edges(psi0, [eX], N)
    pX = probs(psi_X)
    # check if pX equal to p0
    print(f"np.allclose(pX, p0) = {np.allclose(pX, p0)} (should be False)")
    # print max difference between pX and p0
    print(f"max|pX - p0| = {np.max(np.abs(pX - p0)):.6e}")
    # print the states that differ between pX and p0

    print("Differing states:", *np.where(np.abs(pX - p0) > 1e-4)[0])


    print(f"\nProbabilities after single X on edge {eX}:")
    pretty_print_probs(pX)

    # Apply single Z on edge eZ
    eZ = v(0, 0)
    psi_Z = apply_Z_path_edges(psi0, [eZ], N)
    pZ = probs(psi_Z)
    print(f"np.allclose(pZ, p0) = {np.allclose(pZ, p0)} (should be True)")
    print(f"\nProbabilities after single Z on edge {eZ}:")
    pretty_print_probs(pZ)

def main():
    ap = argparse.ArgumentParser(description="Toric code on edges: state prep + visualization.")
    ap.add_argument("--Lx", type=int, default=3)
    ap.add_argument("--Ly", type=int, default=3)
    ap.add_argument("--show-lattice", action="store_true", help="draw bare lattice")
    ap.add_argument("--show-loops", action="store_true", help="draw non-contractible loops")
    ap.add_argument("--braid-demo", action="store_true", help="create m anyons and show braiding phase")
    ap.add_argument("--prob-demo", action="store_true", help="show probs after single X and single Z.")
    ap.add_argument("--label-edges", action="store_true", help="print edge indices on the plot")
    ap.add_argument("--print-probs",action="store_true" , default=0, help="print state probabilities (nonzero only)")
    args = ap.parse_args()

    label_edges = args.label_edges
    print(f"label_edges = {label_edges}")

    Lx, Ly = args.Lx, args.Ly
    N = 2*Lx*Ly
    psi = prepare_ground_state(Lx, Ly)
    if print_probs := args.print_probs:
        p = probs(psi)
        pretty_print_probs(p)


    E0,  = - (np.sum(compute_As_Bp(psi, Lx, Ly), axis=None)),
    As, Bp = compute_As_Bp(psi, Lx, Ly)
    # (The previous line computes twice; keeping simple; below we recompute cleanly.)
    As, Bp = compute_As_Bp(psi, Lx, Ly)
    E0 = - (As.sum() + Bp.sum())
    print(f"Lx x Ly = {Lx} x {Ly} | N={N} qubits | state bytes={psi.nbytes:,}")
    print(f"<A_s>: mean={As.mean():+.6f} min={As.min():+.6f} max={As.max():+.6f}")
    print(f"<B_p>: mean={Bp.mean():+.6f} min={Bp.min():+.6f} max={Bp.max():+.6f}")
    print(f"Ground-state energy E0 = {E0:.6f} (expected {-2*Lx*Ly})")

    if args.show_lattice:
        draw_lattice(Lx, Ly, label_edges=args.label_edges, title="Toric code lattice (edges = qubits)")

    if args.show_loops:
        show_noncontractible_loops(Lx, Ly, label_edges)

    if args.braid_demo:
        braid_demo(Lx, Ly, label_edges)

    if args.prob_demo:
        prob_demo(Lx, Ly)

        


if __name__ == "__main__":
    main()
