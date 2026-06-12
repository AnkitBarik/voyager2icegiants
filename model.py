#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.special import sph_harm_y

class ForwardModel:
    def __init__(self, lmax, r, theta, phi, glm, hlm, idx_g, idx_h):
        self.lmax = lmax
        self.r = r
        self.theta = theta
        self.phi = phi
        self.glm = glm
        self.hlm = hlm
        self.idx_g = idx_g
        self.idx_h = idx_h
        self.npoints = len(r)
        self.ncoeffs = len(glm) + len(hlm)
        self.A = np.zeros((3 * self.npoints, self.ncoeffs))

        for l in range(1, lmax+1):
            fac_l = (1/r)**(l+2)

            for m in range(l+1):
                ylm = sph_harm_y(l, m, theta, phi, diff_n=1)
                dylmdth = ylm[1][..., 0]
                dylmdphi = ylm[1][..., 1]
                ylm = ylm[0]

                if m == 0:
                    Nlm = np.sqrt(4 * np.pi / (2*l+1))
                else:
                    Nlm = np.sqrt(8 * np.pi / (2*l+1)) * (-1)**m

                self.A[0::3, self.idx_g[l, m]] =   np.real(fac_l * (l+1) * Nlm * ylm)
                self.A[1::3, self.idx_g[l, m]] = - np.real(fac_l * Nlm * dylmdth)
                self.A[2::3, self.idx_g[l, m]] = - np.real(fac_l * Nlm / np.sin(theta) * dylmdphi)

                if m > 0:
                    self.A[0::3, self.idx_h[l, m]] =   np.imag(fac_l * (l+1) * Nlm * ylm)
                    self.A[1::3, self.idx_h[l, m]] = - np.imag(fac_l * Nlm * dylmdth)
                    self.A[2::3, self.idx_h[l, m]] = - np.imag(fac_l * Nlm / np.sin(theta) * dylmdphi)

        self.b      = self.A @ np.concatenate([self.glm, self.hlm])
        self.br     = self.b[0::3]
        self.btheta = self.b[1::3]
        self.bphi   = self.b[2::3]

class InverseModel:
    def __init__(self, lmax, r, theta, phi, br, btheta, bphi, smoothing_norm='Br2',
                 sigma=0.5, psi=0.05, a=1, b=0.8, c=0.7, include_external=False, debug=False):

        self.lmax    = lmax
        self.r       = r
        self.theta   = theta
        self.phi     = phi
        self.br      = br
        self.btheta  = btheta
        self.bphi    = bphi
        self.npoints = len(r)
        self.sigma   = sigma
        self.psi     = psi
        self.smoothing_norm = smoothing_norm
        self.a_rad   = a
        self.b_rad   = b
        self.c_rad   = c
        self.include_external = include_external

        count = 0
        idx_g = -np.ones((lmax+1, lmax+1), dtype=int)
        idx_h = -np.ones((lmax+1, lmax+1), dtype=int)

        for l in range(1, lmax+1):
            for m in range(l+1):
                idx_g[l, m] = count
                if debug:
                    print("g(%d,%d), count: %d" %(l,m,count))
                count += 1
                if m > 0:
                    idx_h[l, m] = count
                    if debug:
                        print("h(%d,%d), count: %d" %(l,m,count))
                    count += 1

        self.idx_g = idx_g
        self.idx_h = idx_h

        if self.include_external:
            self.idx_G10 = count
            self.idx_G11 = count + 1
            self.idx_H11 = count + 2
            self.ncoeffs = count + 3
        else:
            self.ncoeffs = count

        # Data vector
        self.b = np.zeros(3 * self.npoints)
        self.A = np.zeros((3 * self.npoints, self.ncoeffs))

        self.b[0::3] = self.br
        self.b[1::3] = self.btheta
        self.b[2::3] = self.bphi

        for l in range(1, lmax+1):
            fac_l = (1/r)**(l+2)

            for m in range(l+1):
                ylm = sph_harm_y(l, m, theta, phi, diff_n=1)
                dylmdth = ylm[1][..., 0]
                dylmdphi = ylm[1][..., 1]
                ylm = ylm[0]

                if m == 0:
                    Nlm = np.sqrt(4 * np.pi / (2*l+1))
                else:
                    Nlm = np.sqrt(8 * np.pi / (2*l+1)) * (-1)**m

                self.A[0::3, self.idx_g[l, m]] =   np.real(fac_l * (l+1) * Nlm * ylm)
                self.A[1::3, self.idx_g[l, m]] = - np.real(fac_l * Nlm * dylmdth)
                self.A[2::3, self.idx_g[l, m]] = - np.real(fac_l * Nlm / np.sin(theta) * dylmdphi)

                if m > 0:
                    self.A[0::3, self.idx_h[l, m]] =   np.imag(fac_l * (l+1) * Nlm * ylm)
                    self.A[1::3, self.idx_h[l, m]] = - np.imag(fac_l * Nlm * dylmdth)
                    self.A[2::3, self.idx_h[l, m]] = - np.imag(fac_l * Nlm / np.sin(theta) * dylmdphi)

        # Fill external field columns
        if self.include_external:
            # G10: external axial dipole
            # Br = -cos(θ), Bθ = sin(θ), Bφ = 0
            self.A[0::3, self.idx_G10] = -np.cos(theta)
            self.A[1::3, self.idx_G10] = np.sin(theta)
            self.A[2::3, self.idx_G10] = 0.0

            # G11: external equatorial dipole (cos φ component)
            # Br = -sin(θ)cos(φ), Bθ = -cos(θ)cos(φ), Bφ = sin(φ)
            self.A[0::3, self.idx_G11] = -np.sin(theta) * np.cos(phi)
            self.A[1::3, self.idx_G11] = -np.cos(theta) * np.cos(phi)
            self.A[2::3, self.idx_G11] = np.sin(phi)

            # H11: external equatorial dipole (sin φ component)
            # Br = -sin(θ)sin(φ), Bθ = -cos(θ)sin(φ), Bφ = -cos(φ)
            self.A[0::3, self.idx_H11] = -np.sin(theta) * np.sin(phi)
            self.A[1::3, self.idx_H11] = -np.cos(theta) * np.sin(phi)
            self.A[2::3, self.idx_H11] = -np.cos(phi)

        self.Ce_inv = self._build_Ce_inv(self.sigma, self.psi)
        self.Lmbda = self._get_lambda(r=1.0)

    def _build_Ce_inv(self, sigma, psi):
        """
        Build block-diagonal Ce_inv per Holme & Bloxham 1995, eq. (4):

        C_e^{-1} = (I σ² + BB^T ψ²) / (σ²(σ² + B²ψ²))
        """

        from scipy.linalg import block_diag

        sigma2 = sigma**2
        psi_rad = np.deg2rad(psi)
        psi2 = psi_rad**2

        B = np.column_stack([self.br, self.btheta, self.bphi])
        B2 = np.sum(B**2, axis=1)

        I3 = np.eye(3)
        blocks = []

        for i in range(self.npoints):
            Bi = B[i, :]
            BBT = np.outer(Bi, Bi)
            denom_i = sigma2 * (sigma2 + B2[i] * psi2)
            numerator = sigma2 * I3 + psi2 * BBT
            blocks.append(numerator / denom_i)

        return block_diag(*blocks)

    def _get_smoothing_norm(self, l, r=1.0):

        if self.smoothing_norm == 'B2':
            return (l + 1) * (1.0/r)**(2*l + 4)
        elif self.smoothing_norm == 'Br2':
            return (l + 1)**2 / (2*l + 1) * (1.0/r)**(2*l + 4)
        elif self.smoothing_norm == 'dBr2':
            return l * (l + 1)**3 / (2*l + 1) * (1.0/r)**(2*l + 6)
        elif self.smoothing_norm == 'energy':
            return (l + 1) / (2*l + 1) * (1.0/r)**(2*l + 1)
        elif self.smoothing_norm == 'ohmic':

            numerator = ( (l + 1) * (2*l + 1) * (2*l + 3) * (2*l + 4) *
                         (self.b_rad - self.c_rad) * self.a_rad**(2*l + 4) )

            denominator = l * (self.b_rad**(2*l + 4) - self.c_rad**(2*l + 4))
            return numerator / denominator
        else:
            return 1.0

    def _get_lambda(self, r=1.0):
        Lmbda = np.zeros((self.ncoeffs, self.ncoeffs))

        for l in range(1, self.lmax+1):
            for m in range(l+1):
                val = self._get_smoothing_norm(l,r=r)
                Lmbda[self.idx_g[l, m], self.idx_g[l, m]] = val
                if m > 0:
                    Lmbda[self.idx_h[l, m], self.idx_h[l, m]] = val

        return Lmbda

    def extract_coeffs(self):
        glm = []
        hlm = []

        for l in range(1, self.lmax + 1):
            for m in range(l + 1):
                glm.append(self.coeffs[self.idx_g[l, m]])
                if m > 0:
                    hlm.append(self.coeffs[self.idx_h[l, m]])
                else:
                    hlm.append(0)

        # Extract external coefficients
        if self.include_external:
            self.G10 = self.coeffs[self.idx_G10]
            self.G11 = self.coeffs[self.idx_G11]
            self.H11 = self.coeffs[self.idx_H11]
            self.glm = np.array(glm)
            self.hlm = np.array(hlm)
        else:
            self.glm = np.array(glm)
            self.hlm = np.array(hlm)

    def solve(self, alpha=0.0, r=1.0):

        AtCeA = self.A.T @ (self.Ce_inv @ self.A)
        AtCeb = self.A.T @ (self.Ce_inv @ self.b)

        LHS = AtCeA + alpha * self.Lmbda
        coeffs = np.linalg.solve(LHS, AtCeb)

        self.coeffs = coeffs
        self.residuals = self.b - self.A @ coeffs
        self.misfit = self.residuals.T @ self.Ce_inv @ self.residuals
        self.norm_value = coeffs.T @ self.Lmbda @ coeffs
        self.extract_coeffs()

    def resolution_matrix(self, alpha=0.0, r=1.0):
        self.Ce_inv = self._build_Ce_inv(self.sigma, self.psi)

        AtCeA = self.A.T @ (self.Ce_inv @ self.A)

        LHS = AtCeA + alpha * self.Lmbda
        inv_LHS = np.linalg.inv(LHS)
        self.R = inv_LHS @ AtCeA

def lcurve_knee(alpha_arr, misfit_arr, norm_arr, trim=2):
    log_misfit = np.log(np.asarray(misfit_arr))
    log_norm = np.log(np.asarray(norm_arr))
    t = np.log(np.asarray(alpha_arr))

    dx = np.gradient(log_misfit, t)
    dy = np.gradient(log_norm, t)

    ddx = np.gradient(dx, t)
    ddy = np.gradient(dy, t)

    curvature = np.abs(dx * ddy - dy * ddx) / np.maximum(
        (dx**2 + dy**2)**1.5,
        1e-300
    )

    curvature[:trim] = -np.inf
    curvature[-trim:] = -np.inf

    idx = int(np.argmax(curvature))
    return idx, alpha_arr[idx], curvature
