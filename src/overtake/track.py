"""

Path geometry helpers (T03) and the corner track with its Frenet frame (T04).

Conventions (see types.py):
- A path is a polyline given by two arrays x, y [m].
- Heading psi is measured counter-clockwise from the x-axis [rad].
- Curvature is signed: positive when the path turns left, negative when it turns right [1/m].
  The course writes the curvature rho; the code calls it kappa.
- Frenet frame: s = arc length along the track centreline, d = lateral offset,
  d > 0 to the left of the centreline.

"""

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np


# ------------------------------------ Path geometry (T03) ---------------------------------------
def _as_path(x, y) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert x, y to two 1-D float arrays of the same length.

    Args:
        x: array-like of float, shape (N,). x coordinates of the points of the path [m].
        y: array-like of float, shape (N,). y coordinates of the points of the path [m].

    Returns:
        (x, y): two np.ndarray of float, shape (N,). The same points, as numpy arrays.

    Raises:
        ValueError: if x and y are not 1-D or do not have the same length.
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be 1-D arrays of the same length.")
    return x, y


def _segments(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectors (dx, dy) and lengths h of the segments joining consecutive points.

    Args:
        x: np.ndarray of float, shape (N,). x coordinates of the points of the path [m].
        y: np.ndarray of float, shape (N,). y coordinates of the points of the path [m].

    Returns:
        dx: np.ndarray of float, shape (N-1,). x component of each segment, x[i+1] - x[i] [m].
        dy: np.ndarray of float, shape (N-1,). y component of each segment, y[i+1] - y[i] [m].
        h:  np.ndarray of float, shape (N-1,). Length of each segment [m].

    Raises:
        ValueError: if two consecutive points coincide (segment of length 0).
    """
    dx, dy = np.diff(x), np.diff(y)
    h = np.hypot(dx, dy)
    if np.any(h == 0):
        raise ValueError("Two consecutive points of the path coincide.")
    return dx, dy, h


def arc_length(x, y) -> np.ndarray:
    """
    Cumulative distance along the polyline [m]; starts at 0, same length as x.

    Args:
        x: array-like of float, shape (N,). x coordinates of the ordered points of a path [m]
           (any path: centreline, racing line, candidate path...).
        y: array-like of float, shape (N,). y coordinates of the same points [m].

    Returns:
        s: np.ndarray of float, shape (N,). Distance travelled along the path from the first
           point to point i [m]; s[0] = 0, s[-1] = total length of the path.
    """
    x, y = _as_path(x, y)
    if len(x) == 0:
        return np.zeros(0)
    return np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))))


def heading(x, y) -> np.ndarray:
    """
    Direction of travel at each point of the polyline [rad], unwrapped (no 2*pi jumps).

    At an interior point the tangent is the one of the circle through the point and its two
    neighbours, so the result is exact on straight lines and on circles, even with uneven spacing.

    Args:
        x: array-like of float, shape (N,), N >= 2. x coordinates of the ordered points of a path [m].
        y: array-like of float, shape (N,). y coordinates of the same points [m].

    Returns:
        psi: np.ndarray of float, shape (N,). Heading at each point [rad], counter-clockwise from
             the x-axis, continuous along the path (it can go beyond pi, e.g. in a hairpin).

    Raises:
        ValueError: if there are fewer than two points, or two consecutive points coincide.
    """
    x, y = _as_path(x, y)
    if len(x) < 2:
        raise ValueError("A heading needs at least two points.")
    dx, dy, h = _segments(x, y)
    seg = np.arctan2(dy, dx)                       # heading of each segment
    psi = np.empty(len(x))
    # Interior points: each neighbouring segment direction is weighted by the length of the other one.
    tx = h[1:] * dx[:-1] / h[:-1] + h[:-1] * dx[1:] / h[1:]
    ty = h[1:] * dy[:-1] / h[:-1] + h[:-1] * dy[1:] / h[1:]
    psi[1:-1] = np.arctan2(ty, tx)
    # End points: on a circle, the chord direction is the mean of the tangent directions at its ends.
    psi[0] = seg[0] if len(x) == 2 else seg[0] + _wrap(seg[0] - psi[1])
    psi[-1] = seg[-1] if len(x) == 2 else seg[-1] + _wrap(seg[-1] - psi[-2])
    return np.unwrap(psi)


def curvature(x, y) -> np.ndarray:
    """
    Signed curvature at each point of the polyline [1/m], positive = turning left.

    Slides 6b: three consecutive waypoints form a triangle with sides a, b, c and area A.
    The circle through the three points has radius abc / (4A), so the curvature is 4A / (abc).
    Using the signed area gives the sign. The two end points copy their neighbour.

    Args:
        x: array-like of float, shape (N,). x coordinates of the ordered points of a path [m].
        y: array-like of float, shape (N,). y coordinates of the same points [m].

    Returns:
        kappa: np.ndarray of float, shape (N,). Signed curvature at each point [1/m]
               (= 1 / radius of the local circle; > 0 left turn, < 0 right turn, 0 straight).
               All zeros if N < 3.

    Raises:
        ValueError: if two consecutive points coincide.
    """
    x, y = _as_path(x, y)
    kappa = np.zeros(len(x))
    if len(x) < 3:
        return kappa
    dx, dy, h = _segments(x, y)
    a, b = h[:-1], h[1:]                                # previous -> point, point -> next
    c = np.hypot(x[2:] - x[:-2], y[2:] - y[:-2])        # previous -> next
    twice_area = dx[:-1] * dy[1:] - dy[:-1] * dx[1:]    # cross product = 2 * signed area
    kappa[1:-1] = 2.0 * twice_area / (a * b * c)
    kappa[0], kappa[-1] = kappa[1], kappa[-2]
    return kappa


def _wrap(angle):
    """
    Wrap an angle to [-pi, pi).

    Args:
        angle: float or np.ndarray of float. Angle(s) [rad], any value.

    Returns:
        np.ndarray of float, same shape as angle. The same angle(s), brought back to [-pi, pi) [rad].
    """
    return (np.asarray(angle) + np.pi) % (2.0 * np.pi) - np.pi


# ------------------------------------ Corner track (T04) ---------------------------------------
class CornerTrack:
    """
    One corner: entry straight -> circular arc -> exit straight, with constant width.

    The centreline starts at the origin and heads along +x. A positive angle is a left turn,
    a negative angle a right turn. The default values are the hairpin of the proposal (R = 15 m, 180 degrees).

    Arrays sampled every ~ds along the centreline: s, cx, cy, psi, kappa.

    Args (constructor):
        l_in:      float. Length of the entry straight [m].
        radius:    float. Radius R of the centreline in the arc [m]; must be > width / 2.
        angle_deg: float. Signed angle of the corner [deg], 0 < |angle| < 360; > 0 left, < 0 right.
        l_out:     float. Length of the exit straight [m].
        width:     float. Track width w, constant [m].
        ds:        float. Approximate spacing of the sampled stations along the centreline [m].

    Attributes:
        angle:        float. Signed corner angle [rad].
        corner_start: float. s at the start of the arc [m].
        corner_end:   float. s at the end of the arc [m].
        length:       float. Total length of the centreline [m].
        s:            np.ndarray, shape (n,). Stations along the centreline, from 0 to length [m].
        cx, cy:       np.ndarray, shape (n,). Global position of the centreline at each station [m].
        psi:          np.ndarray, shape (n,). Heading of the centreline at each station [rad].
        kappa:        np.ndarray, shape (n,). Curvature of the centreline at each station [1/m].

    Raises:
        ValueError: if a length, the radius, the width or ds is not positive, if the angle is 0
                    or not smaller than 360 deg, or if radius <= width / 2.
    """

    def __init__(self, l_in: float = 250.0, radius: float = 15.0, angle_deg: float = 180.0,
                 l_out: float = 150.0, width: float = 12.0, ds: float = 0.5):
        if min(l_in, l_out, radius, width, ds) <= 0:
            raise ValueError("l_in, l_out, radius, width and ds must be positive.")
        if not 0 < abs(angle_deg) < 360:
            raise ValueError("angle_deg must be non-zero and smaller than 360 degrees.")
        if radius <= width / 2:
            raise ValueError("radius must be larger than half the track width.")

        self.l_in, self.radius, self.angle_deg, self.l_out, self.width = l_in, radius, angle_deg, l_out, width
        self.angle = np.deg2rad(angle_deg)              # signed corner angle [rad]
        self._sign = 1.0 if angle_deg > 0 else -1.0     # +1 = left turn, -1 = right turn
        self.corner_start = l_in                        # s at the start of the arc [m]
        self.corner_end = l_in + radius * abs(self.angle)  # s at the end of the arc [m]
        self.length = self.corner_end + l_out           # total length of the centreline [m]

        n = int(round(self.length / ds)) + 1
        self.s = np.linspace(0.0, self.length, n)
        self.cx, self.cy, self.psi, self.kappa = self.at_s(self.s)

    # ---- centreline ----
    def at_s(self, s):
        """
        Centreline x, y, heading and curvature at arc length s (scalar or array).
        Outside [0, length] the straights are extended.

        Args:
            s: float or array-like of float, shape (M,). Arc length(s) along the centreline [m].

        Returns:
            x:     float or np.ndarray (M,). Global x of the centreline point [m].
            y:     float or np.ndarray (M,). Global y of the centreline point [m].
            psi:   float or np.ndarray (M,). Heading of the centreline at s [rad].
            kappa: float or np.ndarray (M,). Curvature of the centreline at s [1/m]
                   (+-1/R in the arc, 0 on the straights).
            Floats if s is a scalar, arrays otherwise.
        """
        s = np.asarray(s, dtype=float)
        before = np.minimum(s, self.corner_start)                    # distance on the entry straight
        phi = np.clip((s - self.corner_start) / self.radius, 0.0, abs(self.angle))  # angle done in the arc
        after = np.maximum(s - self.corner_end, 0.0)                 # distance on the exit straight
        psi = self._sign * phi
        x = before + self.radius * np.sin(phi) + after * np.cos(psi)
        y = self._sign * self.radius * (1.0 - np.cos(phi)) + after * np.sin(psi)
        in_corner = (s >= self.corner_start) & (s <= self.corner_end)
        kappa = np.where(in_corner, self._sign / self.radius, 0.0)
        return _scalars(x, y, psi, kappa)

    # ---- Frenet frame ----
    def from_frenet(self, s, d):
        """
        Frenet (s, d) -> global (x, y): centreline point moved by d along the left normal.

        Args:
            s: float or array-like of float, shape (M,). Arc length along the centreline [m].
            d: float or array-like of float, shape (M,) (or broadcastable with s). Lateral offset
               from the centreline [m], > 0 to the left.

        Returns:
            x: float or np.ndarray (M,). Global x of the point(s) [m].
            y: float or np.ndarray (M,). Global y of the point(s) [m].
            Floats if s and d are scalars, arrays otherwise. With s = track.s and one d per
            station, (x, y) is a path (e.g. a racing line) that the T03 helpers can use.
        """
        cx, cy, psi, _ = self.at_s(s)
        d = np.asarray(d, dtype=float)
        return _scalars(cx - d * np.sin(psi), cy + d * np.cos(psi))

    def to_frenet(self, x, y, s_hint: Optional[float] = None, window: float = 20.0):
        """
        Global (x, y) -> Frenet (s, d): s of the closest centreline point, d > 0 on its left.

        If s_hint is given (for example the s of the previous time step), the closest point is
        searched only in [s_hint - window, s_hint + window]. This removes the ambiguity when
        two parts of the track are close to each other (hairpin).

        Args:
            x: float or array-like of float, shape (M,). Global x of the point(s) [m]. The points
               are projected one by one, so they do not need to form a path (e.g. a car position).
            y: float or array-like of float, shape (M,). Global y of the same point(s) [m].
            s_hint: None, float or array-like (M,). Rough guess of s [m] for each point; None
                    searches the whole track.
            window: float. Half-width of the search interval around s_hint [m].

        Returns:
            s: float or np.ndarray (M,). Arc length of the closest centreline point [m].
            d: float or np.ndarray (M,). Signed lateral offset from the centreline [m], > 0 left.
            Floats if x and y are scalars, arrays otherwise.
        """
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        lo = -np.inf if s_hint is None else np.asarray(s_hint, dtype=float) - window
        hi = np.inf if s_hint is None else np.asarray(s_hint, dtype=float) + window
        cs, ce, R, sig = self.corner_start, self.corner_end, self.radius, self._sign

        # Projection on each of the three pieces, without limits.
        s_in = x                                                    # entry straight lies on the x-axis
        phi = np.arctan2(x - cs, R - sig * y)                       # polar angle around the arc centre
        x1, y1, psi1, _ = self.at_s(ce)
        s_out = ce + (x - x1) * np.cos(psi1) + (y - y1) * np.sin(psi1)

        best_s, best_dist = None, None
        for piece, (a, b) in enumerate(((-np.inf, cs), (cs, ce), (ce, np.inf))):
            a, b = np.maximum(a, lo), np.minimum(b, hi)             # part of the piece inside the window
            if piece == 1:                                          # arc: take the angle nearest to the range
                mid = ((a + b) / 2 - cs) / R
                s_k = cs + R * (mid + _wrap(phi - mid))
            else:
                s_k = s_in if piece == 0 else s_out
            s_k = np.clip(s_k, a, np.maximum(a, b))
            fx, fy, _, _ = self.at_s(s_k)
            dist = np.where(a <= b, np.hypot(x - fx, y - fy), np.inf)
            if best_s is None:
                best_s, best_dist = s_k, dist
            else:
                closer = dist < best_dist
                best_s, best_dist = np.where(closer, s_k, best_s), np.where(closer, dist, best_dist)

        fx, fy, psi, _ = self.at_s(best_s)
        d = -(x - fx) * np.sin(psi) + (y - fy) * np.cos(psi)        # component along the left normal
        return _scalars(best_s, d)

    # ---- track limits ----
    def d_limits(self, car_width: float, margin: float = 0.0) -> Tuple[float, float]:
        """
        Admissible lateral offsets (d_min, d_max) of the car centre, so that the whole car
        stays on the track with a safety margin (proposal eq. 1).

        Args:
            car_width: float. Width of the car [m] (e.g. VehicleParams.width).
            margin:    float. Safety margin between the side of the car and the track edge [m].

        Returns:
            (d_min, d_max): tuple of two floats [m], d_min = -d_max, with
                            d_max = width/2 - car_width/2 - margin.

        Raises:
            ValueError: if the car plus its margins is wider than the track (d_max < 0).
        """
        d_max = self.width / 2 - car_width / 2 - margin
        if d_max < 0:
            raise ValueError("The car (plus margin) is wider than the track.")
        return -d_max, d_max

    def bounds(self):
        """
        Track edges at the sampled stations, for plots: (x_left, y_left), (x_right, y_right).

        Args:
            None (uses self.s and self.width).

        Returns:
            left:  tuple (x_left, y_left) of two np.ndarray, shape (n,). Left edge (d = +width/2) [m].
            right: tuple (x_right, y_right) of two np.ndarray, shape (n,). Right edge (d = -width/2) [m].
        """
        half = self.width / 2
        return self.from_frenet(self.s, half), self.from_frenet(self.s, -half)


def _scalars(*arrays):
    """
    Return Python floats when the inputs were scalars, arrays otherwise.

    Args:
        *arrays: any number of floats or np.ndarray.

    Returns:
        tuple, same length as arrays. Each 0-D value becomes a float, the others an np.ndarray.
    """
    return tuple(float(a) if np.ndim(a) == 0 else np.asarray(a) for a in arrays)
