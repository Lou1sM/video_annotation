# -*- coding: utf-8 -*-


import numbers

import insanity
import torch
import torch.nn as nn

import transformer.decoder as dec
import transformer.encoder as enc
import transformer.transformer as transformer


__author__ = "Patrick Hohenecker"
__copyright__ = (
        "Copyright (c) 2019, Patrick Hohenecker\n"
        "All rights reserved.\n"
        "\n"
        "Redistribution and use in source and binary forms, with or without\n"
        "modification, are permitted provided that the following conditions are met:\n"
        "\n"
        "1. Redistributions of source code must retain the above copyright notice, this\n"
        "   list of conditions and the following disclaimer.\n"
        "2. Redistributions in binary form must reproduce the above copyright notice,\n"
        "   this list of conditions and the following disclaimer in the documentation\n"
        "   and/or other materials provided with the distribution.\n"
        "\n"
        "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS \"AS IS\" AND\n"
        "ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED\n"
        "WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE\n"
        "DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR\n"
        "ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES\n"
        "(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;\n"
        "LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND\n"
        "ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT\n"
        "(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS\n"
        "SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
)
__license__ = "BSD-2-Clause"
__version__ = "2019.1"
__date__ = "27 Jun 2019"
__maintainer__ = "Patrick Hohenecker"
__email__ = "mail@paho.at"
__status__ = "Development"


class RegTransformer(transformer.Transformer):
    """An adaption of the transformer model for general regression tasks."""
    
    def __init__(
            self,
            input_size: int,
            output_size: int,
            num_layers: int = 6,
            num_heads: int = 6,
            dropout_rate: numbers.Real = 0
    ):
        """Creates a new instance of ``Transformer``.

        Args:
            input_size (int): The size of the tokens in any input sequence.
            output_size (int): The size of the tokens in any output sequences to produce.
            num_layers (int, optional): The number of layers to use in both encoder and decoder, which is ``6``, by
                default.
            num_heads (int, optional): The number of attention heads to use, which is ``6``, by default.
            dropout_rate (numbers.Real, optional): The used dropout rate, which is ``0``, by default.
        """
        # sanitize args
        insanity.sanitize_type("input_size", input_size, int)
        insanity.sanitize_range("input_size", input_size, minimum=1)
        insanity.sanitize_type("output_size", output_size, int)
        insanity.sanitize_range("output_size", output_size, minimum=1)
        insanity.sanitize_type("num_layers", num_layers, int)
        insanity.sanitize_range("num_layers", num_layers, minimum=1)
        insanity.sanitize_type("num_heads", num_heads, int)
        insanity.sanitize_range("num_heads", num_heads, minimum=1)
        insanity.sanitize_type("dropout_rate", dropout_rate, numbers.Real)
        dropout_rate = float(dropout_rate)
        insanity.sanitize_range("dropout_rate", dropout_rate, minimum=0, maximum=1, max_inclusive=False)
        
        super().__init__(
                nn.Embedding(1, input_size),  # word_emb -> this is not used by this class
                0,  # pad_index -> does not matter for this class
                output_size,
                max_seq_len=1,  # does not matter for this class
                num_layers=num_layers,
                num_heads=num_heads,
                dim_model=input_size,
                dim_keys=input_size // num_heads,
                dim_values=input_size // num_heads,
                residual_dropout=dropout_rate,
                attention_dropout=dropout_rate
        )
        
        # store input and output size
        self._reg_input_size = input_size
        self._reg_output_size = output_size

        # create linear projection for for mapping targets to the needed input size for the decoder
        self._target_input_projection = nn.Linear(output_size, input_size)
    
    #  METHODS  ########################################################################################################
    
    def forward(self, input_seq: torch.FloatTensor, target: torch.FloatTensor) -> torch.FloatTensor:
        """Runs the Transformer.

        The Transformer expects both an input as well as a target sequence to be provided, and yields a probability
        distribution over all possible output tokens for each position in the target sequence.

        Args:
            input_seq (torch.FloatTensor): The input sequence as (batch-size x input-seq-len x input-size)-tensor.
            target (torch.FloatTensor): The target sequence as (batch-size x target-seq-len x output-size)-tensor.

        Returns:
            torch.FloatTensor: The computed outputs for each position in ``target`` as
            (batch-size x target-seq-len x output-size)-tensor.
        """
        # sanitize args
        if not isinstance(input_seq, torch.Tensor) or input_seq.dtype != torch.float32:
            raise TypeError("<input_seq> has to be a FloatTensor!")
        if input_seq.dim() != 3:
            raise ValueError("<input_seq> has to have 3 dimensions!")
        if input_seq.size(2) != self._reg_input_size:
            raise ValueError("The last dimension of <input_seq> has to be of size {}!".format(self._reg_input_size))
        if not isinstance(target, torch.Tensor) or target.dtype != torch.float32:
            raise TypeError("<target> has to be a FloatTensor!")
        if target.dim() != 3:
            raise ValueError("<target> has to have 3 dimensions!")
        if input_seq.size(0) != target.size(0):
            raise ValueError("The first dimensions (batch) of <input_seq> and <target> have to be of equal size!")
        if target.size(2) != self._reg_output_size:
            raise ValueError("The last dimension of <target> has to be of size {}!".format(self._reg_output_size))
        
        # project input to the needed size
        input_seq = self._input_projection(input_seq)
        
        # run the encoder
        input_seq = self._encoder(input_seq)
        
        # project target to the needed size
        target = self._target_input_projection(target)  # self._input_projection(target)
        
        # run the decoder
        output = self._decoder(input_seq, target)
        
        # project output to the needed size
        output = self._output_projection(output)
        
        return output
    
    def reset_parameters(self) -> None:
        """Resets all trainable parameters of the module."""
        super().reset_parameters()
        self._target_input_projection.reset_parameters()
