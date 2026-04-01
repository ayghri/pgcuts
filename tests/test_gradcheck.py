import torch
from torch.autograd import gradcheck

from pgcuts.hyp2f1.funct import hyp2f1


def test_hyp2f1_gradcheck(device="cpu"):

    a = torch.tensor([0.5], device=device, dtype=torch.float64)
    b = torch.tensor([0.5], device=device, dtype=torch.float64)
    c = torch.tensor([1.5], device=device, dtype=torch.float64)

    z = torch.rand(10, device=device, dtype=torch.float64) * 0.8 + 0.1
    z.requires_grad = True


    def func(z_input):
        return hyp2f1(a, b, c, z_input)

    print("Running gradcheck for hyp2f1 w.r.t z...")
    test = gradcheck(func, (z,), eps=1e-6, atol=1e-4)
    print(f"Gradcheck result: {test}")
    assert test


if __name__ == "__main__":
    try:
        test_hyp2f1_gradcheck()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")

    try:
        test_hyp2f1_gradcheck("cuda")
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
