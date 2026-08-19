# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for FSDP2 uncalled parameter handling during gradient accumulation."""

import unittest
from types import SimpleNamespace

import torch

from verl.utils.fsdp_utils import patch_fsdp2_unsharded_param


class TestFSDP2UncalledParam(unittest.TestCase):
    """Test that patch_fsdp2_unsharded_param guards against AttributeError on uncalled parameters."""

    def setUp(self):
        patch_fsdp2_unsharded_param()

    def test_uncalled_fsdp_param_to_accumulated_grad_safe(self):
        """Simulate an FSDPParam whose _unsharded_param was never initialized because forward bypassed it."""
        try:
            from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
        except ImportError:
            self.skipTest("PyTorch FSDP2 (_fully_shard) is not available")

        mock_fsdp_param = object.__new__(FSDPParam)
        mock_fsdp_param.reduce_dtype = torch.bfloat16
        # Note: _unsharded_param attribute does not exist on this instance

        # Without the patch, this would raise AttributeError: 'FSDPParam' object has no attribute '_unsharded_param'
        mock_fsdp_param.to_accumulated_grad_if_needed()
        mock_fsdp_param.accumulate_unsharded_grad_if_needed()

    def test_called_fsdp_param_to_accumulated_grad_works(self):
        """Verify normal behavior when _unsharded_param is present with gradient."""
        try:
            from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
        except ImportError:
            self.skipTest("PyTorch FSDP2 (_fully_shard) is not available")

        mock_fsdp_param = object.__new__(FSDPParam)
        mock_fsdp_param.reduce_dtype = torch.bfloat16
        mock_fsdp_param._unsharded_param = SimpleNamespace(grad=torch.ones(4, dtype=torch.float32))
        mock_fsdp_param.unsharded_accumulated_grad = None

        mock_fsdp_param.to_accumulated_grad_if_needed()
        self.assertIsNone(mock_fsdp_param._unsharded_param.grad)
        self.assertIsNotNone(mock_fsdp_param.unsharded_accumulated_grad)
        self.assertEqual(mock_fsdp_param.unsharded_accumulated_grad.dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
