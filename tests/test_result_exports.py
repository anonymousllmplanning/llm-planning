"""Synthetic tests for aggregate-only manuscript table conversion."""
import unittest
from scripts.export_paper_results import clean, numeric_tokens, parse_table


class ResultExportTests(unittest.TestCase):
    def test_group_scope_and_footer_are_not_data_rows(self):
        table = r'''\begin{tabular}{lrr}\toprule
Model & Native & Augmented \\\midrule
\multicolumn{3}{l}{\textbf{Closed-model}} \\
\model{GPT-5.4} & 0.104 & \textbf{0.124} \\
\multicolumn{3}{l}{\footnotesize Explanatory note.}
\bottomrule\end{tabular}'''
        rows, _ = parse_table(table, ['model', 'native', 'augmented'])
        self.assertEqual(rows, [{'group': 'Closed-model', 'model': 'GPT-5.4',
                                 'native': '0.104', 'augmented': '0.124'}])

    def test_numeric_precision_and_signs_are_preserved(self):
        raw = r'$\mathbf{+0.052}$ [0.039,0.065]; 1{,}786; 71.33\%'
        value = clean(raw)
        self.assertEqual(value, '+0.052 [0.039,0.065]; 1,786; 71.33%')
        self.assertEqual(numeric_tokens(raw), numeric_tokens(value))

    def test_escaped_ampersand_stays_in_its_cell(self):
        table = r'\begin{tabular}{lr}A & B \\\midrule Text \& Document & 25 \\\end{tabular}'
        rows, _ = parse_table(table, ['name', 'count'])
        self.assertEqual(rows[0]['name'], 'Text & Document')

    def test_wrong_column_count_is_rejected(self):
        table = r'\begin{tabular}{lr}A & B \\\midrule text & 1 & 2 \\\end{tabular}'
        with self.assertRaisesRegex(ValueError, 'Expected 2 cells'):
            parse_table(table, ['name', 'count'])

    def test_unknown_markup_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unrecognized markup'):
            clean(r'\unknown{0.5}')


if __name__ == '__main__':
    unittest.main()
