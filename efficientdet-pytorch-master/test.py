import random
imsize = 1024
w,h= 1024,1024
xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]
print(xc, yc)
indexes = [10] + [random.randint(0, 500) for _ in range(3)]
print(indexes)

x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h

print(x1a, y1a, x2a, y2a)
print(x1b, y1b, x2b, y2b)
padw = x1a - x1b
padh = y1a - y1b
print('-------')
print(padw)
print(padh)
print('-------')
s = 512
x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
print(x1a, y1a, x2a, y2a)
print(x1b, y1b, x2b, y2b)
print('-------')
x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
print(x1a, y1a, x2a, y2a)
print(x1b, y1b, x2b, y2b)
print('-------')

x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
print(x1a, y1a, x2a, y2a)
print(x1b, y1b, x2b, y2b)
