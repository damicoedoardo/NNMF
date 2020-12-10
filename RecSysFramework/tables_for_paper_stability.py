def round_all_digits(number, n_digits):
    return '{:.{prec}f}'.format(number, prec=n_digits)


def tables_for_paper(tables, kind='representations', cutoffs=[5], len_cutoffs_first_metric=5, original_header=[]):
    table = ''
    datasets = list(tables.keys())
    header_datasets = ''
    for dataset in datasets:
        header_datasets += ' & \\multicolumn{' + \
            str(len(cutoffs)*2) + '}{c|}{\\textbf{' + dataset + '}}'
    multirow_algs = '2' if len(cutoffs) == 1 else '3'
    format_table = 'c'+('|'+'c'*len(cutoffs))*len(datasets) * \
        2 if len(cutoffs) > 1 else 'c'+('|'+'c'*len(cutoffs)*2)*len(datasets)
    table += '  \\begin{table*} \
                \\centering \
                \\caption{Stability of ' + kind + '} \
                \\begin{tabular}{' + format_table + '} \
                \\toprule \
                \\multirow{' + multirow_algs + '}{*}{\\textbf{Algorithm}}' + header_datasets + ' \\\\ '
    if kind == 'recommendations':
        if len(cutoffs) > 1:
            table += (' & \\multicolumn{' + str(len(cutoffs)) + '}{c|}{Jaccard} & \\multicolumn{' + str(
                len(cutoffs)) + '}{c|}{RBO}')*len(datasets) + ' \\\\'
        else:
            table += (' & \\multicolumn{' + str(len(cutoffs)) + '}{c}{Jaccard} & \\multicolumn{' + str(
                len(cutoffs)) + '}{c|}{RBO}')*len(datasets) + ' \\\\'
    elif kind == 'representations':
        if len(cutoffs) > 1:
            table += (' & \\multicolumn{' + str(len(cutoffs)) + '}{c|}{Items} & \\multicolumn{' + str(
                len(cutoffs)) + '}{c|}{Users}')*len(datasets) + ' \\\\'
        else:
            table += (' & \\multicolumn{' + str(len(cutoffs)) + '}{c}{Items} & \\multicolumn{' + str(
                len(cutoffs)) + '}{c|}{Users}')*len(datasets) + ' \\\\'

    if len(cutoffs) > 1:
        header_cutoffs = ''
        for cutoff in cutoffs:
            header_cutoffs += ' & {}'.format(original_header[cutoff-1])
        for cutoff in cutoffs:
            header_cutoffs += ' & {}'.format(
                original_header[cutoff+len_cutoffs_first_metric-1])
        header_cutoffs *= len(datasets)
        table += header_cutoffs + ' \\\\'
    table += ' \\midrule '

    alg_score = {}
    for dataset in datasets:
        text = tables[dataset]
        rows = text.split('\\\\')
        for row in rows:
            if len(row) > 0:
                tokens = [s.replace('\\midrule', '').strip()
                          for s in row.split('&')]
                alg = tokens[0]
                if alg not in alg_score:
                    alg_score[alg] = alg
                for cutoff in cutoffs:
                    alg_score[alg] += ' & {}'.format(
                        round_score_number(tokens[cutoff]))
                for cutoff in cutoffs:
                    alg_score[alg] += ' & {}'.format(round_score_number(
                        tokens[cutoff+len_cutoffs_first_metric]))
    for idx, alg in enumerate(alg_score.keys()):
        table += '\n ' + \
            alg_score[alg] + ' \\\\ \\midrule' if idx % 2 != 0 else '\n ' + \
            alg_score[alg] + ' \\\\'

    table += '\\end{tabular} \\label{table:stability_' + \
        kind + '} \\end{table*}'
    return table


def round_score_number(s):
    s_ = s.replace('\\textbf', '').replace(
        '\\underline', '').replace('{', '').replace('}', '')
    s__ = str(round_all_digits(float(s_), 2))
    return s.replace(s_, s__)


if __name__ == "__main__":
    # Not opt params
    #     print(tables_for_paper({'\\lastfm': 'BPRMF & 0.7306 & 0.7414 & 0.7327 & 0.7184 & 0.7031 & 0.6695 & 0.7000 & 0.7219 & 0.7452 & 0.7723 \\\\  \
    #     BPR NNMF & \\underline{0.8296} & \\underline{0.8407} & \\underline{0.8516} & \\underline{0.8604} & \\underline{0.8700} & \\underline{0.8236} & \\underline{0.8429} & \\underline{0.8611} & \\underline{0.8827} & \\underline{0.9028} \\\\ \\midrule \
    #     FunkSVD & 0.6648 & 0.7051 & 0.7254 & 0.7343 & 0.7313 & 0.6103 & 0.6463 & 0.6791 & 0.7098 & 0.7384 \\\\  \
    #     Funk NNMF & \\underline{0.8042} & \\underline{0.8302} & \\underline{0.8505} & \\underline{0.8702} & \\underline{0.8857} & \\underline{0.7985} & \\underline{0.8311} & \\underline{0.8566} & \\underline{0.8824} & \\underline{0.9088} \\\\ \\midrule \
    #     ProbMF & 0.6159 & 0.6296 & 0.6225 & 0.6039 & 0.5850 & 0.5603 & 0.5875 & 0.6063 & 0.6270 & 0.6477 \\\\  \
    #     Prob NNMF & \\textbf{\\underline{0.8339}} & \\textbf{\\underline{0.8565}} & \\textbf{\\underline{0.8711}} & \\textbf{\\underline{0.8869}} & \\textbf{\\underline{0.9006}} & \\textbf{\\underline{0.8285}} & \\textbf{\\underline{0.8550}} & \\textbf{\\underline{0.8758}} & \\textbf{\\underline{0.8954}} & \\textbf{\\underline{0.9198}} \\\\',
    #                             '\\movielensom': 'BPRMF & 0.7017 & 0.7507 & 0.7856 & 0.8164 & 0.8445 & 0.7030 & 0.7445 & 0.7733 & 0.8025 & 0.8288 \\\\ \
    #                             BPR NNMF & \\textbf{\\underline{0.9019}} & \\textbf{\\underline{0.9174}} & \\textbf{\\underline{0.9296}} & \\textbf{\\underline{0.9391}} & \\textbf{\\underline{0.9474}} & \\textbf{\\underline{0.8203}} & \\textbf{\\underline{0.8449}} & \\textbf{\\underline{0.8602}} & \\textbf{\\underline{0.8754}} & \\textbf{\\underline{0.8891}} \\\\ \\midrule\
    #                             FunkSVD & 0.3614 & 0.4412 & 0.5057 & 0.5748 & \\textbf{\\underline{0.6442}} & 0.3346 & 0.4061 & 0.4645 & 0.5262 & 0.5915 \\\\ \
    #                             Funk NNMF & \\textbf{\\underline{0.4535}} & \\textbf{\\underline{0.5015}} & \\textbf{\\underline{0.5447}} & \\textbf{\\underline{0.5902}} & 0.6347 & \\textbf{\\underline{0.4305}} & \\textbf{\\underline{0.4779}} & \\textbf{\\underline{0.5188}} & \\textbf{\\underline{0.5597}} & \\textbf{\\underline{0.6025}} \\\\ \\midrule\
    #                             ProbMF & 0.5484 & 0.6244 & 0.6804 & 0.7337 & 0.7838 & 0.4679 & 0.5241 & 0.5676 & 0.6118 & 0.6550 \\\\ \
    #                             Prob NNMF & \\textbf{\\underline{0.8455}} & \\textbf{\\underline{0.8722}} & \\textbf{\\underline{0.8901}} & \\textbf{\\underline{0.9045}} & \\textbf{\\underline{0.9179}} & \\textbf{\\underline{0.7901}} & \\textbf{\\underline{0.8138}} & \\textbf{\\underline{0.8354}} & \\textbf{\\underline{0.8561}} & \\textbf{\\underline{0.8748}} \\\\',
    #                             '\\bookcrossing': 'BPRMF & 0.4699 & 0.4724 & 0.4738 & 0.4767 & 0.4754 & 0.3154 & 0.3237 & 0.3465 & 0.3759 & 0.4052 \\\\ \
    #                                 BPR NNMF & \\textbf{\\underline{0.6758}} & \\textbf{\\underline{0.6702}} & \\textbf{\\underline{0.6674}} & \\textbf{\\underline{0.6658}} & \\textbf{\\underline{0.6579}} & \\textbf{\\underline{0.5186}} & \\textbf{\\underline{0.4810}} & \\textbf{\\underline{0.4836}} & \\textbf{\\underline{0.5040}} & \\textbf{\\underline{0.5338}} \\\\ \\midrule\
    #                                 FunkSVD & 0.1714 & 0.1595 & 0.1468 & 0.1360 & 0.1304 & 0.1346 & 0.0990 & 0.0848 & 0.0794 & 0.0783 \\\\ \
    #                                 Funk NNMF & \\textbf{\\underline{0.3000}} & \\textbf{\\underline{0.2343}} & \\textbf{\\underline{0.2113}} & \\textbf{\\underline{0.2017}} & \\textbf{\\underline{0.1896}} & \\textbf{\\underline{0.3224}} & \\textbf{\\underline{0.2414}} & \\textbf{\\underline{0.2032}} & \\textbf{\\underline{0.1800}} & \\textbf{\\underline{0.1706}} \\\\ \\midrule\
    #                                 ProbMF & 0.3778 & 0.2602 & 0.2257 & 0.2305 & 0.2658 & 0.2581 & 0.2454 & 0.2567 & 0.2769 & 0.2987 \\\\ \
    #                                 Prob NNMF & \\textbf{\\underline{0.6542}} & \\textbf{\\underline{0.6623}} & \\textbf{\\underline{0.6689}} & \\textbf{\\underline{0.6764}} & \\textbf{\\underline{0.6676}} & \\textbf{\\underline{0.4501}} & \\textbf{\\underline{0.4226}} & \\textbf{\\underline{0.4313}} & \\textbf{\\underline{0.4547}} & \\textbf{\\underline{0.4850}} \\\\',
    #                             '\\pinterest': 'BPRMF & 0.4458 & 0.5285 & 0.5986 & 0.6747 & 0.7303 & 0.2961 & 0.3409 & 0.3821 & 0.4288 & 0.4797 \\\\ \
    #                                 BPR NNMF & \\textbf{\\underline{0.8108}} & \\textbf{\\underline{0.8518}} & \\textbf{\\underline{0.8829}} & \\textbf{\\underline{0.9124}} & \\textbf{\\underline{0.9320}} & \\textbf{\\underline{0.7025}} & \\textbf{\\underline{0.7390}} & \\textbf{\\underline{0.7689}} & \\textbf{\\underline{0.8007}} & \\textbf{\\underline{0.8335}} \\\\ \\midrule\
    #                                 FunkSVD & 0.3421 & 0.4343 & 0.5176 & 0.6006 & 0.6589 & 0.1940 & 0.2350 & 0.2764 & 0.3279 & 0.3897 \\\\ \
    #                                 Funk NNMF & \\textbf{\\underline{0.7097}} & \\textbf{\\underline{0.7667}} & \\textbf{\\underline{0.8130}} & \\textbf{\\underline{0.8569}} & \\textbf{\\underline{0.8907}} & \\textbf{\\underline{0.6029}} & \\textbf{\\underline{0.6433}} & \\textbf{\\underline{0.6768}} & \\textbf{\\underline{0.7121}} & \\textbf{\\underline{0.7511}} \\\\ \\midrule\
    #                                 ProbMF & 0.3173 & 0.3987 & 0.4735 & 0.5559 & 0.6116 & 0.1947 & 0.2287 & 0.2654 & 0.3135 & 0.3715 \\\\ \
    #                                 Prob NNMF & \\textbf{\\underline{0.7305}} & \\textbf{\\underline{0.7824}} & \\textbf{\\underline{0.8260}} & \\textbf{\\underline{0.8710}} & \\textbf{\\underline{0.9016}} & \\textbf{\\underline{0.6039}} & \\textbf{\\underline{0.6396}} & \\textbf{\\underline{0.6736}} & \\textbf{\\underline{0.7119}} & \\textbf{\\underline{0.7552}} \\\\',
    #                             '\\epinions': 'BPRMF & 0.5365 & 0.5185 & 0.5076 & 0.5100 & 0.5257 & 0.4526 & 0.4343 & 0.4394 & 0.4584 & 0.4871 \\\\ \
    #                                 BPR NNMF & \\textbf{\\underline{0.6574}} & \\textbf{\\underline{0.6143}} & \\textbf{\\underline{0.5967}} & \\textbf{\\underline{0.5967}} & \\textbf{\\underline{0.6103}} & \\textbf{\\underline{0.5715}} & \\textbf{\\underline{0.5415}} & \\textbf{\\underline{0.5430}} & \\textbf{\\underline{0.5600}} & \\textbf{\\underline{0.5867}} \\\\ \\midrule\
    #                                 FunkSVD & \\textbf{\\underline{0.1385}} & \\textbf{\\underline{0.1112}} & \\textbf{\\underline{0.0960}} & \\textbf{\\underline{0.0860}} & \\textbf{\\underline{0.0808}} & 0.0317 & 0.0299 & 0.0306 & 0.0334 & 0.0394 \\\\ \
    #                                 Funk NNMF & 0.1144 & 0.0683 & 0.0520 & 0.0437 & 0.0418 & \\textbf{\\underline{0.1258}} & \\textbf{\\underline{0.0788}} & \\textbf{\\underline{0.0591}} & \\textbf{\\underline{0.0479}} & \\textbf{\\underline{0.0434}} \\\\ \\midrule\
    #                                 ProbMF & 0.1638 & 0.1435 & 0.1476 & 0.1646 & 0.1903 & 0.0590 & 0.0642 & 0.0738 & 0.0873 & 0.1043 \\\\ \
    #                                 Prob NNMF & \\textbf{\\underline{0.5778}} & \\textbf{\\underline{0.5096}} & \\textbf{\\underline{0.4513}} & \\textbf{\\underline{0.4012}} & \\textbf{\\underline{0.3688}} & \\textbf{\\underline{0.3095}} & \\textbf{\\underline{0.2527}} & \\textbf{\\underline{0.2432}} & \\textbf{\\underline{0.2512}} & \\textbf{\\underline{0.2723}} \\\\',
    #                             '\\citulike': 'BPRMF & 0.5712 & 0.5971 & 0.6164 & 0.6343 & 0.6428 & 0.5844 & 0.5962 & 0.6052 & 0.6139 & 0.6094 \\\\ \
    #                                 BPR NNMF & \\textbf{\\underline{0.7586}} & \\textbf{\\underline{0.7528}} & \\textbf{\\underline{0.7581}} & \\textbf{\\underline{0.7711}} & \\textbf{\\underline{0.7847}} & \\textbf{\\underline{0.7616}} & \\textbf{\\underline{0.7607}} & \\textbf{\\underline{0.7719}} & \\textbf{\\underline{0.7886}} & \\textbf{\\underline{0.7994}} \\\\ \\midrule\
    #                                 FunkSVD & 0.4760 & 0.4826 & 0.4783 & 0.4766 & 0.4703 & 0.4815 & 0.4835 & 0.4646 & 0.4268 & 0.3567 \\\\ \
    #                                 Funk NNMF & \\textbf{\\underline{0.6964}} & \\textbf{\\underline{0.6986}} & \\textbf{\\underline{0.7133}} & \\textbf{\\underline{0.7368}} & \\textbf{\\underline{0.7600}} & \\textbf{\\underline{0.7160}} & \\textbf{\\underline{0.7299}} & \\textbf{\\underline{0.7514}} & \\textbf{\\underline{0.7788}} & \\textbf{\\underline{0.7942}} \\\\ \\midrule\
    #                                 ProbMF & 0.6028 & 0.6069 & 0.6073 & 0.6164 & 0.6183 & 0.3961 & 0.4003 & 0.3953 & 0.3815 & 0.3471 \\\\ \
    #                                 Prob NNMF & \\textbf{\\underline{0.7567}} & \\textbf{\\underline{0.7328}} & \\textbf{\\underline{0.7309}} & \\textbf{\\underline{0.7437}} & \\textbf{\\underline{0.7548}} & \\textbf{\\underline{0.6607}} & \\textbf{\\underline{0.6357}} & \\textbf{\\underline{0.6364}} & \\textbf{\\underline{0.6383}} & \\textbf{\\underline{0.6138}} \\\\', },
    #                            kind='representations', cutoffs=[1], len_cutoffs_first_metric=5, original_header=[10, 25, 50, 100, 200, 10, 25, 50, 100, 200]))

    #     print(tables_for_paper({'\\lastfm': 'BPRMF & 0.7253 & 0.6975 & 0.7244 & 0.7555 & 0.7703 & 0.7959 & 0.8260 \\\\  \
    # BPR NNMF & \\underline{0.9013} & \\underline{0.8762} & \\underline{0.8847} & \\underline{0.8973} & \\underline{0.9137} & \\underline{0.9229} & \\underline{0.9333} \\\\ \\midrule \
    # FunkSVD & 0.7385 & 0.7228 & 0.7476 & 0.7854 & 0.7850 & 0.8108 & 0.8423 \\\\  \
    # Funk NNMF & \\underline{0.9325} & \\textbf{\\underline{0.9160}} & \\underline{0.9144} & \\underline{0.9298} & \\underline{0.9413} & \\underline{0.9446} & \\underline{0.9524} \\\\ \\midrule \
    # ProbMF & 0.6798 & 0.6183 & 0.6414 & 0.6702 & 0.7150 & 0.7387 & 0.7688 \\\\  \
    # Prob NNMF & \\textbf{\\underline{0.9403}} & \\underline{0.9111} & \\textbf{\\underline{0.9161}} & \\textbf{\\underline{0.9304}} & \\textbf{\\underline{0.9420}} & \\textbf{\\underline{0.9460}} & \\textbf{\\underline{0.9539}} \\\\',
    #                             '\\movielensom': 'BPRMF & 0.7988 & 0.7769 & 0.8015 & 0.8365 & 0.8327 & 0.8535 & 0.8808 \\\\  \
    #                             BPR NNMF & \\textbf{\\underline{0.9212}} & \\textbf{\\underline{0.9081}} & \\textbf{\\underline{0.9157}} & \\textbf{\\underline{0.9314}} & \\textbf{\\underline{0.9349}} & \\textbf{\\underline{0.9426}} & \\textbf{\\underline{0.9526}} \\\\ \\midrule \
    #                             FunkSVD & 0.6311 & 0.5980 & 0.6487 & \\textbf{\\underline{0.7135}} & 0.6792 & 0.7168 & \\textbf{\\underline{0.7699}} \\\\  \
    #                             Funk NNMF & \\textbf{\\underline{0.6803}} & \\textbf{\\underline{0.6323}} & \\textbf{\\underline{0.6669}} & 0.7132 & \\textbf{\\underline{0.6997}} & \\textbf{\\underline{0.7289}} & 0.7688 \\\\ \\midrule \
    #                             ProbMF & 0.6327 & 0.6046 & 0.6464 & 0.6984 & 0.6899 & 0.7247 & 0.7708 \\\\  \
    #                             Prob NNMF & \\textbf{\\underline{0.8737}} & \\textbf{\\underline{0.8576}} & \\textbf{\\underline{0.8720}} & \\textbf{\\underline{0.8948}} & \\textbf{\\underline{0.8960}} & \\textbf{\\underline{0.9086}} & \\textbf{\\underline{0.9256}} \\\\',
    #                             '\\bookcrossing': 'BPRMF & 0.2775 & 0.2824 & 0.3147 & 0.3638 & 0.3494 & 0.3900 & 0.4466 \\\\  \
    # BPR NNMF & \\textbf{\\underline{0.6170}} & \\textbf{\\underline{0.5827}} & \\textbf{\\underline{0.6026}} & \\textbf{\\underline{0.6356}} & \\textbf{\\underline{0.6595}} & \\textbf{\\underline{0.6874}} & \\textbf{\\underline{0.7224}} \\\\ \\midrule \
    # FunkSVD & 0.0527 & 0.0532 & 0.0592 & 0.0715 & 0.0735 & 0.0845 & 0.1020 \\\\  \
    # Funk NNMF & \\textbf{\\underline{0.2040}} & \\textbf{\\underline{0.1495}} & \\textbf{\\underline{0.1395}} & \\textbf{\\underline{0.1331}} & \\textbf{\\underline{0.1873}} & \\textbf{\\underline{0.1764}} & \\textbf{\\underline{0.1636}} \\\\ \\midrule \
    # ProbMF & 0.2775 & 0.2310 & 0.2406 & 0.2611 & 0.3181 & 0.3371 & 0.3645 \\\\  \
    # Prob NNMF & \\textbf{\\underline{0.7439}} & \\textbf{\\underline{0.6735}} & \\textbf{\\underline{0.6724}} & \\textbf{\\underline{0.6750}} & \\textbf{\\underline{0.7466}} & \\textbf{\\underline{0.7596}} & \\textbf{\\underline{0.7727}} \\\\',
    #                             '\\pinterest': 'BPRMF & 0.5171 & 0.4993 & 0.5262 & 0.5706 & 0.5851 & 0.6184 & 0.6619 \\\\ \
    # BPR NNMF & \\textbf{\\underline{0.9461}} & \\textbf{\\underline{0.9266}} & \\textbf{\\underline{0.9264}} & \\textbf{\\underline{0.9328}} & \\textbf{\\underline{0.9523}} & \\textbf{\\underline{0.9546}} & \\textbf{\\underline{0.9589}} \\\\ \\midrule \
    # FunkSVD & 0.2623 & 0.2809 & 0.3281 & 0.4094 & 0.3467 & 0.3975 & 0.4754 \\\\  \
    # Funk NNMF & \\textbf{\\underline{0.8048}} & \\textbf{\\underline{0.7939}} & \\textbf{\\underline{0.8092}} & \\textbf{\\underline{0.8353}} & \\textbf{\\underline{0.8299}} & \\textbf{\\underline{0.8448}} & \\textbf{\\underline{0.8658}} \\\\ \\midrule \
    # ProbMF & 0.3797 & 0.3933 & 0.4391 & 0.5079 & 0.4712 & 0.5179 & 0.5835 \\\\  \
    # Prob NNMF & \\textbf{\\underline{0.9196}} & \\textbf{\\underline{0.8928}} & \\textbf{\\underline{0.8917}} & \\textbf{\\underline{0.9015}} & \\textbf{\\underline{0.9278}} & \\textbf{\\underline{0.9329}} & \\textbf{\\underline{0.9388}} \\\\',
    #                             '\\epinions': 'BPRMF & 0.5264 & 0.5008 & 0.5294 & 0.5689 & 0.5895 & 0.6229 & 0.6653 \\\\  \
    # BPR NNMF & \\textbf{\\underline{0.6755}} & \\textbf{\\underline{0.6258}} & \\textbf{\\underline{0.6393}} & \\textbf{\\underline{0.6689}} & \\textbf{\\underline{0.7124}} & \\textbf{\\underline{0.7312}} & \\textbf{\\underline{0.7572}} \\\\ \\midrule \
    # FunkSVD & 0.0185 & \\textbf{\\underline{0.0274}} & \\textbf{\\underline{0.0366}} & \\textbf{\\underline{0.0525}} & \\textbf{\\underline{0.0342}} & \\textbf{\\underline{0.0463}} & \\textbf{\\underline{0.0676}} \\\\  \
    # Funk NNMF & \\textbf{\\underline{0.0189}} & 0.0172 & 0.0212 & 0.0302 & 0.0250 & 0.0301 & 0.0411 \\\\ \\midrule \
    # ProbMF & 0.2470 & 0.2627 & 0.3104 & 0.3203 & 0.3229 & 0.3735 & 0.4272 \\\\  \
    # Prob NNMF & \\textbf{\\underline{0.4277}} & \\textbf{\\underline{0.4558}} & \\textbf{\\underline{0.4789}} & \\textbf{\\underline{0.4983}} & \\textbf{\\underline{0.5091}} & \\textbf{\\underline{0.5501}} & \\textbf{\\underline{0.5918}} \\\\',
    #                             '\\citulike': 'BPRMF & 0.5272 & 0.4930 & 0.5140 & 0.5450 & 0.5885 & 0.6176 & 0.6542 \\\\  \
    # BPR NNMF & \\textbf{\\underline{0.7622}} & \\textbf{\\underline{0.7271}} & \\textbf{\\underline{0.7399}} & \\textbf{\\underline{0.7555}} & \\textbf{\\underline{0.7967}} & \\textbf{\\underline{0.8137}} & \\textbf{\\underline{0.8338}} \\\\ \\midrule \
    # FunkSVD & 0.3045 & 0.3110 & 0.3346 & 0.3613 & 0.3851 & 0.4230 & 0.4673 \\\\  \
    # Funk NNMF & \\textbf{\\underline{0.7252}} & \\textbf{\\underline{0.6700}} & \\textbf{\\underline{0.6874}} & \\textbf{\\underline{0.7172}} & \\textbf{\\underline{0.7542}} & \\textbf{\\underline{0.7735}} & \\textbf{\\underline{0.7993}} \\\\ \\midrule \
    # ProbMF & 0.3877 & 0.3706 & 0.3944 & 0.4234 & 0.4621 & 0.4960 & 0.5369 \\\\  \
    # Prob NNMF & \\textbf{\\underline{0.6242}} & \\textbf{\\underline{0.5932}} & \\textbf{\\underline{0.6120}} & \\textbf{\\underline{0.6423}} & \\textbf{\\underline{0.6757}} & \\textbf{\\underline{0.7011}} & \\textbf{\\underline{0.7334}} \\\\', },
    #                            kind='recommendations', cutoffs=[2], len_cutoffs_first_metric=3, original_header=[1, 5, 10, 25, 5, 10, 25]))

    # Opt params
    print(tables_for_paper({'\\lastfm': 'BPRMF & 0.7306 & 0.7414 & 0.7327 & 0.7184 & 0.7031 & 0.6695 & 0.7000 & 0.7219 & 0.7452 & 0.7723 \\\\  \
    BPR NNMF  & \\textbf{\\underline{0.8151}} & \\underline{0.8301} & \\underline{0.8399} & \\underline{0.8513} & \\underline{0.8594} & \\textbf{\\underline{0.8707}} & \\textbf{\\underline{0.8839}} & \\textbf{\\underline{0.8945}} & \\textbf{\\underline{0.9060}} & \\textbf{\\underline{0.9179}} \\\\ \\midrule \
    FunkSVD & 0.6648 & 0.7051 & 0.7254 & 0.7343 & 0.7313 & 0.6103 & 0.6463 & 0.6791 & 0.7098 & 0.7384 \\\\  \
    Funk NNMF & \\underline{0.8084} & \\textbf{\\underline{0.8401}} & \\textbf{\\underline{0.8595}} & \\textbf{\\underline{0.8701}} & \\textbf{\\underline{0.8721}} & \\underline{0.8256} & \\underline{0.8453} & \\underline{0.8617} & \\underline{0.8769} & \\underline{0.8899} \\\\ \\midrule \
    ProbMF & 0.6159 & 0.6296 & 0.6225 & 0.6039 & 0.5850 & 0.5603 & 0.5875 & 0.6063 & 0.6270 & 0.6477 \\\\  \
    Prob NNMF & \\underline{0.7606} & \\underline{0.7682} & \\underline{0.7761} & \\underline{0.7811} & \\underline{0.7867} & \\underline{0.7827} & \\underline{0.8080} & \\underline{0.8248} & \\underline{0.8426} & \\underline{0.8576} \\\\',
                            '\\movielensom': 'BPRMF & 0.7017 & 0.7507 & 0.7856 & 0.8164 & 0.8445 & 0.7030 & 0.7445 & 0.7733 & 0.8025 & 0.8288 \\\\ \
                            BPR NNMF & \\textbf{\\underline{0.9280}} & \\textbf{\\underline{0.9413}} & \\textbf{\\underline{0.9506}} & \\textbf{\\underline{0.9592}} & \\textbf{\\underline{0.9661}} & \\textbf{\\underline{0.9122}} & \\textbf{\\underline{0.9271}} & \\textbf{\\underline{0.9373}} & \\textbf{\\underline{0.9469}} & \\textbf{\\underline{0.9547}} \\\\ \\midrule\
                            FunkSVD & 0.3614 & 0.4412 & 0.5057 & 0.5748 & \\textbf{\\underline{0.6442}} & 0.3346 & 0.4061 & 0.4645 & 0.5262 & 0.5915 \\\\ \
                            Funk NNMF & \\textbf{\\underline{0.9451}} & \\textbf{\\underline{0.9540}} & \\textbf{\\underline{0.9616}} & \\textbf{\\underline{0.9681}} & \\textbf{\\underline{0.9736}} & \\textbf{\\underline{0.9195}} & \\textbf{\\underline{0.9336}} & \\textbf{\\underline{0.9425}} & \\textbf{\\underline{0.9507}} & \\textbf{\\underline{0.9575}} \\\\ \\midrule\
                            ProbMF & 0.5484 & 0.6244 & 0.6804 & 0.7337 & 0.7838 & 0.4679 & 0.5241 & 0.5676 & 0.6118 & 0.6550 \\\\ \
                            Prob NNMF & \\textbf{\\underline{0.8121}} & \\textbf{\\underline{0.8397}} & \\textbf{\\underline{0.8561}} & \\textbf{\\underline{0.8666}} & \\textbf{\\underline{0.8754}} & \\textbf{\\underline{0.6841}} & \\textbf{\\underline{0.7148}} & \\textbf{\\underline{0.7396}} & \\textbf{\\underline{0.7617}} & \\textbf{\\underline{0.7846}} \\\\',
                            '\\bookcrossing': 'BPRMF & 0.4699 & 0.4724 & 0.4738 & 0.4767 & 0.4754 & 0.3154 & 0.3237 & 0.3465 & 0.3759 & 0.4052 \\\\ \
                                BPR NNMF & \\textbf{\\underline{0.5518}} & \\textbf{\\underline{0.5180}} & \\textbf{\\underline{0.5163}} & \\textbf{\\underline{0.5279}} & \\textbf{\\underline{0.5486}} & \\textbf{\\underline{0.5668}} & \\textbf{\\underline{0.5251}} & \\textbf{\\underline{0.5212}} & \\textbf{\\underline{0.5371}} & \\textbf{\\underline{0.5632}} \\\\ \\midrule\
                                FunkSVD & 0.1714 & 0.1595 & 0.1468 & 0.1360 & 0.1304 & 0.1346 & 0.0990 & 0.0848 & 0.0794 & 0.0783 \\\\ \
                                Funk NNMF & \\textbf{\\underline{0.4214}} & \\textbf{\\underline{0.3169}} & \\textbf{\\underline{0.2768}} & \\textbf{\\underline{0.2616}} & \\textbf{\\underline{0.2647}} & \\textbf{\\underline{0.2487}} & \\textbf{\\underline{0.1852}} & \\textbf{\\underline{0.1708}} & \\textbf{\\underline{0.1717}} & \\textbf{\\underline{0.1779}} \\\\ \\midrule\
                                ProbMF & 0.3778 & 0.2602 & 0.2257 & 0.2305 & 0.2658 & 0.2581 & 0.2454 & 0.2567 & 0.2769 & 0.2987 \\\\ \
                                Prob NNMF & \\textbf{\\underline{0.5712}} & \\textbf{\\underline{0.5484}} & \\textbf{\\underline{0.5503}} & \\textbf{\\underline{0.5685}} & \\textbf{\\underline{0.5983}} & \\textbf{\\underline{0.6173}} & \\textbf{\\underline{0.5619}} & \\textbf{\\underline{0.5365}} & \\textbf{\\underline{0.5337}} & \\textbf{\\underline{0.5431}} \\\\',
                            '\\pinterest': 'BPRMF & 0.4458 & 0.5285 & 0.5986 & 0.6747 & 0.7303 & 0.2961 & 0.3409 & 0.3821 & 0.4288 & 0.4797 \\\\ \
                                    BPR NNMF & \\textbf{\\underline{0.6560}} & \\textbf{\\underline{0.7168}} & \\textbf{\\underline{0.7613}} & \\textbf{\\underline{0.8058}} & \\textbf{\\underline{0.8219}} & \\textbf{\\underline{0.5109}} & \\textbf{\\underline{0.5513}} & \\textbf{\\underline{0.5865}} & \\textbf{\\underline{0.6259}} & \\textbf{\\underline{0.6689}} \\\\ \\midrule\
                                    FunkSVD & 0.3421 & 0.4343 & 0.5176 & 0.6006 & 0.6589 & 0.1940 & 0.2350 & 0.2764 & 0.3279 & 0.3897 \\\\ \
                                    Funk NNMF & \\textbf{\\underline{0.6824}} & \\textbf{\\underline{0.7495}} & \\textbf{\\underline{0.8011}} & \\textbf{\\underline{0.8422}} & \\textbf{\\underline{0.8554}} & \\textbf{\\underline{0.5760}} & \\textbf{\\underline{0.6205}} & \\textbf{\\underline{0.6607}} & \\textbf{\\underline{0.7040}} & \\textbf{\\underline{0.7503}} \\\\ \\midrule\
                                    ProbMF & 0.3173 & 0.3987 & 0.4735 & 0.5559 & 0.6116 & 0.1947 & 0.2287 & 0.2654 & 0.3135 & 0.3715 \\\\ \
                                    Prob NNMF & \\textbf{\\underline{0.6304}} & \\textbf{\\underline{0.6970}} & \\textbf{\\underline{0.7490}} & \\textbf{\\underline{0.8022}} & \\textbf{\\underline{0.8296}} & \\textbf{\\underline{0.4642}} & \\textbf{\\underline{0.5122}} & \\textbf{\\underline{0.5538}} & \\textbf{\\underline{0.6014}} & \\textbf{\\underline{0.6557}} \\\\',
                            '\\epinions': 'BPRMF & 0.5365 & 0.5185 & 0.5076 & 0.5100 & 0.5257 & 0.4526 & 0.4343 & 0.4394 & 0.4584 & 0.4871 \\\\ \
                                BPR NNMF & \\textbf{\\underline{0.6829}} & \\textbf{\\underline{0.6529}} & \\textbf{\\underline{0.6464}} & \\textbf{\\underline{0.6554}} & \\textbf{\\underline{0.6746}} & \\textbf{\\underline{0.5824}} & \\textbf{\\underline{0.5721}} & \\textbf{\\underline{0.5866}} & \\textbf{\\underline{0.6120}} & \\textbf{\\underline{0.6443}} \\\\ \\midrule\
                                FunkSVD & \\textbf{\\underline{0.1385}} & \\textbf{\\underline{0.1112}} & \\textbf{\\underline{0.0960}} & \\textbf{\\underline{0.0860}} & \\textbf{\\underline{0.0808}} & 0.0317 & 0.0299 & 0.0306 & 0.0334 & 0.0394 \\\\ \
                                Funk NNMF & \\textbf{\\underline{0.5989}} & \\textbf{\\underline{0.5454}} & \\textbf{\\underline{0.5023}} & \\textbf{\\underline{0.4660}} & \\textbf{\\underline{0.4431}} & \\textbf{\\underline{0.3647}} & \\textbf{\\underline{0.3279}} & \\textbf{\\underline{0.3286}} & \\textbf{\\underline{0.3434}} & \\textbf{\\underline{0.3670}} \\\\ \\midrule\
                                ProbMF & 0.1638 & 0.1435 & 0.1476 & 0.1646 & 0.1903 & 0.0590 & 0.0642 & 0.0738 & 0.0873 & 0.1043 \\\\ \
                                Prob NNMF & \\textbf{\\underline{0.7139}} & \\textbf{\\underline{0.6472}} & \\textbf{\\underline{0.5993}} & \\textbf{\\underline{0.5666}} & \\textbf{\\underline{0.5539}} & \\textbf{\\underline{0.5709}} & \\textbf{\\underline{0.4698}} & \\textbf{\\underline{0.4325}} & \\textbf{\\underline{0.4199}} & \\textbf{\\underline{0.4222}} \\\\',
                            '\\citulike': 'BPRMF & 0.5712 & 0.5971 & 0.6164 & 0.6343 & 0.6428 & 0.5844 & 0.5962 & 0.6052 & 0.6139 & 0.6094 \\\\ \
                                BPR NNMF & \\textbf{\\underline{0.6886}} & \\textbf{\\underline{0.6803}} & \\textbf{\\underline{0.6891}} & \\textbf{\\underline{0.7056}} & \\textbf{\\underline{0.7210}} & \\textbf{\\underline{0.6940}} & \\textbf{\\underline{0.6916}} & \\textbf{\\underline{0.7002}} & \\textbf{\\underline{0.7183}} & \\textbf{\\underline{0.7257}} \\\\ \\midrule\
                                FunkSVD & 0.4760 & 0.4826 & 0.4783 & 0.4766 & 0.4703 & 0.4815 & 0.4835 & 0.4646 & 0.4268 & 0.3567 \\\\ \
                                Funk NNMF & \\textbf{\\underline{0.6430}} & \\textbf{\\underline{0.6194}} & \\textbf{\\underline{0.6214}} & \\textbf{\\underline{0.6313}} & \\textbf{\\underline{0.6320}} & \\textbf{\\underline{0.6075}} & \\textbf{\\underline{0.5699}} & \\textbf{\\underline{0.5436}} & \\textbf{\\underline{0.4997}} & \\textbf{\\underline{0.4269}} \\\\ \\midrule\
                                ProbMF & 0.6028 & 0.6069 & 0.6073 & 0.6164 & 0.6183 & 0.3961 & 0.4003 & 0.3953 & 0.3815 & 0.3471 \\\\ \
                                Prob NNMF & \\textbf{\\underline{0.7503}} & \\textbf{\\underline{0.7379}} & \\textbf{\\underline{0.7390}} & \\textbf{\\underline{0.7505}} & \\textbf{\\underline{0.7592}} & \\textbf{\\underline{0.7060}} & \\textbf{\\underline{0.7173}} & \\textbf{\\underline{0.7306}} & \\textbf{\\underline{0.7419}} & \\textbf{\\underline{0.7292}} \\\\', },
                           kind='representations', cutoffs=[1], len_cutoffs_first_metric=5, original_header=[10, 25, 50, 100, 200, 10, 25, 50, 100, 200]))

    print(tables_for_paper({'\\lastfm': 'BPRMF & 0.7253 & 0.6975 & 0.7244 & 0.7555 & 0.7703 & 0.7959 & 0.8260 \\\\  \
    BPR NNMF & \\underline{0.8851} & \\underline{0.8573} & \\underline{0.8643} & \\underline{0.8805} & \\underline{0.9026} & \\underline{0.9112} & \\underline{0.9226} \\\\ \\midrule \
    FunkSVD & 0.7385 & 0.7228 & 0.7476 & 0.7854 & 0.7850 & 0.8108 & 0.8423 \\\\  \
    Funk NNMF & \\textbf{\\underline{0.9065}} & \\textbf{\\underline{0.8705}} & \\textbf{\\underline{0.8851}} & \\textbf{\\underline{0.8980}} & \\textbf{\\underline{0.9143}} & \\textbf{\\underline{0.9221}} & \\textbf{\\underline{0.9330}} \\\\ \\midrule \
    ProbMF & 0.6798 & 0.6183 & 0.6414 & 0.6702 & 0.7150 & 0.7387 & 0.7688 \\\\  \
    Prob NNMF & \\underline{0.8291} & \\underline{0.7930} & \\underline{0.8082} & \\underline{0.8239} & \\underline{0.8527} & \\underline{0.8685} & \\underline{0.8848} \\\\',
                            '\\movielensom': 'BPRMF & 0.7988 & 0.7769 & 0.8015 & 0.8365 & 0.8327 & 0.8535 & 0.8808 \\\\  \
                                BPR NNMF & \\textbf{\\underline{0.9592}} & \\textbf{\\underline{0.9497}} & \\textbf{\\underline{0.9516}} & \\textbf{\\underline{0.9584}} & \\textbf{\\underline{0.9659}} & \\textbf{\\underline{0.9692}} & \\textbf{\\underline{0.9737}} \\\\ \\midrule \
                                FunkSVD & 0.6311 & 0.5980 & 0.6487 & \\textbf{\\underline{0.7135}} & 0.6792 & 0.7168 & \\textbf{\\underline{0.7699}} \\\\  \
                                Funk NNMF & \\textbf{\\underline{0.9627}} & \\textbf{\\underline{0.9534}} & \\textbf{\\underline{0.9572}} & \\textbf{\\underline{0.9657}} & \\textbf{\\underline{0.9690}} & \\textbf{\\underline{0.9720}} & \\textbf{\\underline{0.9768}} \\\\ \\midrule \
                                ProbMF & 0.6327 & 0.6046 & 0.6464 & 0.6984 & 0.6899 & 0.7247 & 0.7708 \\\\  \
                                Prob NNMF & \\textbf{\\underline{0.7956}} & \\textbf{\\underline{0.7636}} & \\textbf{\\underline{0.7864}} & \\textbf{\\underline{0.8217}} & \\textbf{\\underline{0.8274}} & \\textbf{\\underline{0.8463}} & \\textbf{\\underline{0.8722}} \\\\',
                            '\\bookcrossing': 'BPRMF & 0.2775 & 0.2824 & 0.3147 & 0.3638 & 0.3494 & 0.3900 & 0.4466 \\\\  \
    BPR NNMF & \\textbf{\\underline{0.5809}} & \\textbf{\\underline{0.5239}} & \\textbf{\\underline{0.5377}} & \\textbf{\\underline{0.5659}} & \\textbf{\\underline{0.6279}} & \\textbf{\\underline{0.6492}} & \\textbf{\\underline{0.6786}} \\\\ \\midrule \
    FunkSVD & 0.0527 & 0.0532 & 0.0592 & 0.0715 & 0.0735 & 0.0845 & 0.1020 \\\\  \
    Funk NNMF & \\textbf{\\underline{0.1781}} & \\textbf{\\underline{0.1757}} & \\textbf{\\underline{0.1899}} & \\textbf{\\underline{0.2026}} & \\textbf{\\underline{0.2277}} & \\textbf{\\underline{0.2534}} & \\textbf{\\underline{0.2830}} \\\\ \\midrule \
    ProbMF & 0.2775 & 0.2310 & 0.2406 & 0.2611 & 0.3181 & 0.3371 & 0.3645 \\\\  \
    Prob NNMF & \\textbf{\\underline{0.6664}} & \\textbf{\\underline{0.6138}} & \\textbf{\\underline{0.6260}} & \\textbf{\\underline{0.6421}} & \\textbf{\\underline{0.7028}} & \\textbf{\\underline{0.7217}} & \\textbf{\\underline{0.7448}} \\\\',
                            '\\pinterest': 'BPRMF & 0.5171 & 0.4993 & 0.5262 & 0.5706 & 0.5851 & 0.6184 & 0.6619 \\\\ \
    BPR NNMF &  \\textbf{\\underline{0.7373}} & \\textbf{\\underline{0.6889}} & \\textbf{\\underline{0.7097}} & \\textbf{\\underline{0.7446}} & \\textbf{\\underline{0.7709}} & \\textbf{\\underline{0.7913}} & \\textbf{\\underline{0.8191}} \\\\ \\midrule \
    FunkSVD & 0.2623 & 0.2809 & 0.3281 & 0.4094 & 0.3467 & 0.3975 & 0.4754 \\\\  \
    Funk NNMF & \\textbf{\\underline{0.7781}} & \\textbf{\\underline{0.7505}} & \\textbf{\\underline{0.7751}} & \\textbf{\\underline{0.8146}} & \\textbf{\\underline{0.8146}} & \\textbf{\\underline{0.8358}} & \\textbf{\\underline{0.8646}} \\\\ \\midrule \
    ProbMF & 0.3797 & 0.3933 & 0.4391 & 0.5079 & 0.4712 & 0.5179 & 0.5835 \\\\  \
    Prob NNMF &  \\textbf{\\underline{0.8338}} & \\textbf{\\underline{0.7764}} & \\textbf{\\underline{0.7900}} & \\textbf{\\underline{0.8125}} & \\textbf{\\underline{0.8443}} & \\textbf{\\underline{0.8563}} & \\textbf{\\underline{0.8739}} \\\\',
                            '\\epinions': 'BPRMF & 0.5264 & 0.5008 & 0.5294 & 0.5689 & 0.5895 & 0.6229 & 0.6653 \\\\  \
    BPR NNMF & \\textbf{\\underline{0.7214}} & \\textbf{\\underline{0.7000}} & \\textbf{\\underline{0.7197}} & \\textbf{\\underline{0.7457}} & \\textbf{\\underline{0.7675}} & \\textbf{\\underline{0.7913}} & \\textbf{\\underline{0.8189}} \\\\ \\midrule \
    FunkSVD & 0.0185 & \\textbf{\\underline{0.0274}} & \\textbf{\\underline{0.0366}} & \\textbf{\\underline{0.0525}} & \\textbf{\\underline{0.0342}} & \\textbf{\\underline{0.0463}} & \\textbf{\\underline{0.0676}} \\\\  \
    Funk NNMF & \\textbf{\\underline{0.5383}} & \\textbf{\\underline{0.5058}} & \\textbf{\\underline{0.5224}} & \\textbf{\\underline{0.5449}} & \\textbf{\\underline{0.5956}} & \\textbf{\\underline{0.6237}} & \\textbf{\\underline{0.6548}} \\\\ \\midrule \
    ProbMF & 0.2470 & 0.2627 & 0.3104 & 0.3203 & 0.3229 & 0.3735 & 0.4272 \\\\  \
    Prob NNMF & \\textbf{\\underline{0.6194}} & \\textbf{\\underline{0.5673}} & \\textbf{\\underline{0.5725}} & \\textbf{\\underline{0.5784}} & \\textbf{\\underline{0.6616}} & \\textbf{\\underline{0.6819}} & \\textbf{\\underline{0.7023}} \\\\',
                            '\\citulike': 'BPRMF & 0.5272 & 0.4930 & 0.5140 & 0.5450 & 0.5885 & 0.6176 & 0.6542 \\\\  \
    BPR NNMF & \\textbf{\\underline{0.6979}} & \\textbf{\\underline{0.6452}} & \\textbf{\\underline{0.6594}} & \\textbf{\\underline{0.6819}} & \\textbf{\\underline{0.7368}} & \\textbf{\\underline{0.7560}} & \\textbf{\\underline{0.7797}} \\\\ \\midrule \
    FunkSVD & 0.3045 & 0.3110 & 0.3346 & 0.3613 & 0.3851 & 0.4230 & 0.4673 \\\\  \
    Funk NNMF & \\textbf{\\underline{0.5507}} & \\textbf{\\underline{0.5140}} & \\textbf{\\underline{0.5313}} & \\textbf{\\underline{0.5533}} & \\textbf{\\underline{0.6125}} & \\textbf{\\underline{0.6410}} & \\textbf{\\underline{0.6728}} \\\\ \\midrule \
    ProbMF & 0.3877 & 0.3706 & 0.3944 & 0.4234 & 0.4621 & 0.4960 & 0.5369 \\\\  \
    Prob NNMF & \\textbf{\\underline{0.6920}} & \\textbf{\\underline{0.6587}} & \\textbf{\\underline{0.6756}} & \\textbf{\\underline{0.7053}} & \\textbf{\\underline{0.7389}} & \\textbf{\\underline{0.7626}} & \\textbf{\\underline{0.7921}} \\\\', },
                           kind='recommendations', cutoffs=[2], len_cutoffs_first_metric=3, original_header=[1, 5, 10, 25, 5, 10, 25]))
