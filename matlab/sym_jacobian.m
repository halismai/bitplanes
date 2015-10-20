p = sym('p', [1 8]);
syms x y w real;
syms s c1 c2 real;
syms Ix Iy real;
T = [s 0 -s*c1; 0 s -s*c2; 0 0 1];

H = sl3_hat(p);

Jh = [];
for i = 1 : length(p)
  Jh = [Jh inv(T)*jacobian(H*T*[x;y;1], p(i))];
end

xw = [x/w; y/w];
Jp = [jacobian(xw, x), jacobian(xw, y), jacobian(xw,w)];

Jw  = simplify(subs(Jp, 'w', 1) * Jh);
J = simplify([Ix Iy] * Jw);

