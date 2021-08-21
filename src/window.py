# window.py
#
# Copyright 2021 julian
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from gi.repository import Gtk

@Gtk.Template(resource_path='/com/schelb/summarizing/window.ui')
class SummarizingTextWindow(Gtk.ApplicationWindow):
    __gtype_name__ = 'SummarizingTextWindow'

    input_text = Gtk.Template.Child("input_text_orig")
    output_summary = Gtk.Template.Child("output_summary")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        ### init input field with sampel text
        text ="""
            The US has "passed the peak" on new coronavirus cases, President Donald Trump said and predicted that some states would reopen this month.
            The US has over 637,000 confirmed Covid-19 cases and over 30,826 deaths, the highest for any country in the world.
            At the daily White House coronavirus briefing on Wednesday, Trump said new guidelines to reopen the country would be announced on Thursday after he speaks to governors.
            "We'll be the comeback kids, all of us," he said. "We want to get our country back."
            The Trump administration has previously fixed May 1 as a possible date to reopen the world's largest economy, but the president said some states may be able to return to normalcy earlier than that.
            """
        self.input_text_textbuffer = self.input_text.get_buffer()
        self.input_text_textbuffer.set_text(text)


        ### load model
        self.model = T5ForConditionalGeneration.from_pretrained('/app/share/model/', cache_dir=None)
        self.tokenizer = T5Tokenizer.from_pretrained('/app/share/model/', cache_dir=None)
        self.device = torch.device('cpu')



    @Gtk.Template.Callback()
    def onButtonPressed(self, button):

        ### read from input field
        (iter_first, iter_last) = self.input_text_textbuffer.get_bounds()
        text = self.input_text_textbuffer.get_text(iter_first, iter_last, False)

        ### preprocessing
        preprocess_text = text.strip().replace("\n","")
        t5_prepared_Text = "summarize: "+preprocess_text
        tokenized_text = self.tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(self.device)


        ### summmarize
        summary_ids = self.model.generate(tokenized_text,
                                            num_beams=4,
                                            no_repeat_ngram_size=2,
                                            min_length=30,
                                            max_length=100,
                                            early_stopping=True)

        ### set output
        output = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        self.textbuffer = self.output_summary.get_buffer()
        self.textbuffer.set_text(output)
