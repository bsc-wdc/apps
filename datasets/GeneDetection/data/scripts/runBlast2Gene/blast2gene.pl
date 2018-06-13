#!/usr/bin/perl
#
#   blast2gene.pl [-mincov 0.7]  *.bp  [pagen|all(default)]  [-geneonly]  [-noscore]  [-gff]  >  *.bl2g
#
#   Recommended conditions for tblastn:
#     -F  F       filter OFF
#     -e  0.001   E-value cutoff
#     -M  PAM70   (cf. www.ncbi.nlm.nih.gov/BLAST/matrix_info.html#matrix)
#     -G  10
#     -E  1
#
#
#   ver.7
#     - get minimum coverage from ARGV
#     - GENE or FRAGMENT based on minimum coverage
#                                                   30/05/2003    Mikita Suyama
#   ver.8
#     - typo of valuable is fixed [posov -> pepov]
#     - new scoring: 'd2score'
#         <10    30
#        11-30   20
#        31-80   13
#        81-160  10
#       161-1000  5
#     - new overlap condition
#        jlen/qlenj >= 0.7 && pepov/klen >= 0.9  ->  -5
#        klen/qlenk >= 0.7 && pepov/jlen >= 0.9  ->  -5
#     - posext
#         0  ->  -10
#                                                   02/06/2003
#
#   ver.9                                           23/07/2003
#     - tblastn -> bp -> -3,-2,-1,+1,+2,+3  (column 14)   already works
#       blastn  -> bp -> Plus (14)  Plus/Minus (15)       modified to work
#       blastp  -> bp -> 0-1.000 (14)                     modified to work
#
#   ver.10                                          25/07/2003
#     - overlapping exon --> separate genes
#       some conditions are changed for overlapping fragments
#
#   ver.11                                          08/08/2003
#     - page select (default: all)
#       page number is written at the end of QID-SID block
#         ## 1 / 8     (page 1 [out of 8 pages])
#
#     - -geneonly option
#       only report GENEs
#
#   ver.12                                          15/08/2003
#     - before:  ////  ->  cut genes
#       new:     ////  ->  cut genes if terminal; don't cut if internal
#
#     - In the case of genome overlap, if both e-values are the same,
#       then take the higher id%.
#
#     - '-noscore' option                           16/12/2003
#
#   ver.13                                          29/12/2003
#     - -gff option
#
#   ver.14
#     - not sort by col[8] but min(col[8],col[10])
#                                                   28/01/2004
#
#     - bug fix at "remove fragments overlap in genomic position"
#           typo: some datj should be datk
#                                                   20/04/2004
#
#   ver.15
#     - to prevent fragmentation, delete all fragments overlapping
#       more than $gnmovercut but but the one with significant difference
#       with the 2nd hit (e value difference more than 1e-10).
#
#     - new scoring: 'd2score'
#         <10    30
#        11-30   20
#        31-80   13
#        81-160  10
#       161-1000  5
#      1001-5000  1
#      5000-     null
#                                                   20/04/2004
#     - new scoring: 'd2score'
#         <10    30
#        11-20   25
#        21-30   20
#        31-80   13
#        81-160  10
#       161-1000  5
#      1001-5000  1
#      5000-     null
#
#     - $gnmovercut 30 -> 35
#
#     - if (breaklines contain <= 3) then take the best one
#       elsif (breaklines contain >3) ckeck e-val diff to the 2nd best
#
#                                                   29/04/2004
#     - bug fix (datj -> dat)
#                                                   03/08/2004
#     - $overlaplen >= $gnmovercut
#         -> 
#       $overlaplen >= $gnmovercut ||
#       ($pl1 <= $pk1 && $pk2 <= $pl2) ||       frag-k within frag-l
#       ($pk1 <= $pl1 && $pl2 <= $pk2)          frag-l within frag-k
#
#                                                   17/08/2004
#   ver.16
#     - Dr. Gabor Toth kindly reported that there is a bug in the script:
#       "It (Illegal division by zero at ./blast2gene.pl line 657.) appears
#        if there are only DELETED HSPs for a certain query--hit pair."
#       Now it is fixed. Thank you, Dr. Toth.
#                                                   03/12/2004
#
#   ver.17
#     - Dr. David Garcia kindly reported the following bug:
#         datj -> dat
#       in "remove fragments overlap in genomic position"
#       Thank you, David.
#                                                   31/07/2006
#

$| = 1;

$gnmovercut = 35;
$lookup = 5;
$maxintron = 1000000;

#-----------#
#  DEFAULT  #
#-----------#
$mincov = 0.7;        #   minimum coverage for a gene


#------------#
#  Get ARGV  #
#------------#

if ($#ARGV == -1) {
    print STDERR "blast2gene (v17)\n\n";
    print STDERR "Usage:  blast2gene.pl  [-mincov  0.7]  *.bp  [pagen|all(default)]  [-geneonly]  [-noscore]  [-gff]  >  *.bl2g\n\n";
    exit;
} else {
    $getmc = 0;
    undef($bpfile);
    $pagen = "all";
    $geneonly = "off";
    $noscore = "off";
    $gff = "off";
    foreach $i (0..$#ARGV) {
        if ($ARGV[$i] eq '-mincov') {
            $getmc = 1;
        } elsif ($ARGV[$i] =~ /^-mincov(\S+)/) {
            $mincov = $1;
            $getmc = 0;
        } elsif ($getmc) {
            $mincov = $ARGV[$i];
            $getmc = 0;
        } elsif ($ARGV[$i] eq "-geneonly") {
            $geneonly = "on";
        } elsif ($ARGV[$i] eq "-noscore") {
            $noscore = "on";
        } elsif ($ARGV[$i] eq "-gff") {
            $gff = "on";
        } elsif (!$bpfile) {
            $bpfile = $ARGV[$i];
        } elsif ($ARGV[$i] =~ /^\d+$/ || $ARGV[$i] =~ /^all$/) {
            $pagen = $ARGV[$i];
        } else {
            print STDERR "blast2gene (v13)\n\n";
            print STDERR "Usage:  blast2gene.pl  [-mincov  0.7]  *.bp  [pagen|all(default)]  [-geneonly]  [-noscore]  >  *.bl2g\n\n";
            exit;
        }
    }
}




undef(@entries);
undef($tmpsid);
undef($presid);
$n = 0;
open(BPFILE, "< $bpfile") || die "Can't open $bpfile";
while (<BPFILE>) {
    chomp;
    if (/^##/) {
        ++$n;
        undef($tmpsid);
        undef($presid);
    } elsif (/^\/\//) {
        # NOTHING TO DO, JUST SKIP (*.bp should not have this line.)
    } elsif (/No hits found/) {
        --$n;
    } else {
        @dat = split(/\s+/, $_);
        $tmpsid = $dat[1];
        if ($tmpsid ne $presid && $presid) {
            ++$n;
        }
        $entries[$n] .= "$_\n";
        $presid = $tmpsid;
    }
}
close(BPFILE);


#----------------------------#
# fragment distance -> score #
#   D <=10         30
#   10< D <=20     25
#   20< D <=30     20
#   30< D <=80     13
#   80< D <=160    10
#  160< D <=1000    5
# 1000< D <=5000    1
# 5000< D           0 (null)
#----------------------------#

undef(%d2score);
foreach $i (0..5000) {
    if ($i <= 10) {
        $d2score{$i} = 30;
    } elsif ($i <= 20) {
        $d2score{$i} = 25;
    } elsif ($i <= 30) {
        $d2score{$i} = 20;
    } elsif ($i <= 80) {
        $d2score{$i} = 13;
    } elsif ($i <= 160) {
        $d2score{$i} = 10;
    } elsif ($i <= 1000) {
        $d2score{$i} =  5;
    } else {
        $d2score{$i} =  1;
    }
}


#---------------------#
# foreach qid-sid set #
#---------------------#

$totpair = $#entries + 1;

foreach $i (0..$#entries) {
  if ($pagen eq "all" || $pagen == $i + 1) {
    $tmppair = $i + 1;

    @lines = split(/\n/, $entries[$i]);

    #--------------------------#
    # sort by genomic position # 
    #--------------------------#
    local(@datakeys);
    foreach (@lines) {
        @dat = split(/\s+/, $_);
        if ($dat[8] < $dat[10]) {
            push(@datakeys, $dat[8]);
        } else {
            push(@datakeys, $dat[10]);
        }
#        push(@datakeys, (split(/\s+/))[8]);
    }
    sub bydatakeys { $datakeys[$a] <=> $datakeys[$b]; }
    @sortdata = @lines[sort bydatakeys $[..$#lines];

    #----------------------------------------------------------#
    # remove fragments overlap in genomic position             #
    # (take lower E (if both E are the same, take higher id%)) #
    #----------------------------------------------------------#

    ## make @breaklines                                          ##
    ##   (group them according to the vertical line in bp2plot)  ##

    undef(@breaklines);
    $nbreak = 0;
    foreach $j (0..$#sortdata) {
        if ($j == 0) {
            $breaklines[$nbreak] .= "$sortdata[$j]\n";
            @dat = split(/\s+/, $sortdata[$j]);
            if ($dat[8] < $dat[10]) {
                $groupminpos = $dat[8];
                $groupmaxpos = $dat[10];
            } else {
                $groupminpos = $dat[10];
                $groupmaxpos = $dat[8];
            }
        } else {
            @dat = split(/\s+/, $sortdata[$j]);
            if ($dat[8] < $dat[10]) {
                $tmpminpos = $dat[8];
                $tmpmaxpos = $dat[10];
            } else {
                $tmpminpos = $dat[10];
                $tmpmaxpos = $dat[8];
            }
            $overgnm = & overlen($groupminpos, $groupmaxpos, $tmpminpos, $tmpmaxpos);
            if ($overgnm >= 1) {
                $breaklines[$nbreak] .= "$sortdata[$j]\n";
                if ($tmpminpos < $groupminpos) {
                    $groupminpos = $tmpminpos;
                }
                if ($tmpmaxpos > $groupmaxpos) {
                    $groupmaxpos = $tmpmaxpos;
                }
            } else {
                ++$nbreak;
                $breaklines[$nbreak] .= "$sortdata[$j]\n";
                $groupminpos = $tmpminpos;
                $groupmaxpos = $tmpmaxpos;
            }
        }
    }

    ## foreach @breaklines, select the most significant one (or none),      ##
    ## if they overlap more than $gnmovercut.                               ##
    ## if tmp breaklines contains > 3 segments, check the 2nd best          ##

    undef(@deleted);
    undef(@newsorted);
    foreach $j (0..$nbreak) {
        @tmplines = split(/\n/, $breaklines[$j]);

        undef(@fragalive);
        foreach $k (0..$#tmplines) {
            $fragalive[$k] = 1;
        }

        foreach $k (0..$#tmplines - 1) {
            @datk = split(/\s+/, $tmplines[$k]);
            if ($datk[8] < $datk[10]) {
                $pk1 = $datk[8];
                $pk2 = $datk[10];
            } else {
                $pk1 = $datk[10];
                $pk2 = $datk[8];
            }
            foreach $l ($k + 1..$#tmplines) {
                @datl = split(/\s+/, $tmplines[$l]);
                if ($datl[8] < $datl[10]) {
                    $pl1 = $datl[8];
                    $pl2 = $datl[10];
                } else {
                    $pl1 = $datl[10];
                    $pl2 = $datl[8];
                }

                $overlaplen = & overlen($pk1, $pk2, $pl1, $pl2);
                if ($#tmplines + 1 > 3) {

                    ##  >= $gnmovercut OR
                    ##  frag-k within frag-l OR
                    ##  frag-l within frag-k

                    if ($overlaplen >= $gnmovercut ||
                       ($pl1 <= $pk1 && $pk2 <= $pl2) ||
                       ($pk1 <= $pl1 && $pl2 <= $pk2)) {
                        if ($datk[3] <= $datl[3]) {
                            if ($datl[3] == 0) {
                                $fragalive[$k] = 0;
                                $fragalive[$l] = 0;
                            } elsif ($datk[3] / $datl[3] <= 1e-10) {
                                $fragalive[$l] = 0;
                            } else {
                                $fragalive[$k] = 0;
                                $fragalive[$l] = 0;
                            }
                        } else {
                            if ($datl[3] / $datk[3] <= 1e-10) {
                                $fragalive[$k] = 0;
                            } else {
                                $fragalive[$k] = 0;
                                $fragalive[$l] = 0;
                            }
                        }
                    }
                } else {

                    ##  >= $gnmovercut OR
                    ##  frag-k within frag-l OR
                    ##  frag-l within frag-k

                    if ($overlaplen >= $gnmovercut ||
                        ($pl1 <= $pk1 && $pk2 <= $pl2) ||
                        ($pk1 <= $pl1 && $pl2 <= $pk2)) {
                        if ($datk[3] < $datl[3]) {
                            $fragalive[$l] = 0;
                        } elsif ($datl[3] < $datk[3]) {
                            $fragalive[$k] = 0;
                        } elsif ($datk[3] == $datl[3]) {
                            $fragalive[$l] = 0;
                        }
                    }
                }
            }
        }

        foreach $k (0..$#tmplines) {
            if ($fragalive[$k]) {
                push(@newsorted, $tmplines[$k]);
            } else {
                push(@deleted, $tmplines[$k]);
            }
        }
    }
    @sortdata = @newsorted;

    
    #-------------------#
    # make score matrix #
    #-------------------#

    undef(%score);
    $testmode = 0;
    foreach $j (0..$#sortdata) {
        printf "%-3d  ", $j if ($testmode);    ## TEST  ##
        @datj = split(/\s+/, $sortdata[$j]);
        if ($datj[8] < $datj[10]) {
            $dirj = "for";
        } else {
            $dirj = "rev";
        }
        $posj1 = $datj[4];
        $posj2 = $datj[6];
        $jlen = $posj2 - $posj1 + 1;
        $qlenj = $datj[7];

        $enddata = $j + $lookup;
        $enddata = $#sortdata if ($enddata > $#sortdata);
        foreach $k ($j+1..$enddata) {

            #--------------------------------------------------------#
            # if (same direction) {
            #   if (pos extension > 0) {
            #     if (j-coverage >= 0.7 ||
            #         (jlen/qlenj >= 0.7 && $pepov / $klen >= 0.9) ||
            #         (klen/qlenk >= 0.7 && $pepov / $jlen >= 0.9)) {
            #       score = -5
            #     } elsif (genome dist > maxintron) {
            #       score = -8
            #     } else {
            #       score = d2score(abs[posdif]) - skippenalty)
            #         skippenalty: -log10[Eval]
            #     }
            #   } else {
            #    score = -10
            # } else {
            #   score = -20
            # }
            #--------------------------------------------------------#
            @datk = split(/\s+/, $sortdata[$k]);
            if ($datk[8] < $datk[10]) {
                $dirk = "for";
            } else {
                $dirk = "rev";
            }
            $posk1 = $datk[4];
            $posk2 = $datk[6];
            $klen = $posk2 - $posk1 + 1;
            $evalk = $datk[3];
            $evalk = 1e-200 if ($evalk < 1e-200);
            $qlenk = $datk[7];
            undef($posdif);

            if ($dirj eq $dirk) {
                $pepov = overlen($posj1, $posj2, $posk1, $posk2);
                if ($dirj eq "for") {
                    $posdif = $posk1 - ($posj2 + 1);
                    $posext = $posk2 - $posj2;
                    $gnmdist = $datk[10] - $datj[8] - 1;
                } else {
                    $posdif = $posj1 - ($posk2 + 1);
                    $posext = $posj1 - $posk1;
                    $gnmdist = $datk[8] - $datj[10] - 1;
                }

                if ($pepov / $klen >= 0.8 && $pepov / $jlen >= 0.8) {        ##  "//////"  ##
                    ##  IF these are the terminal exons
                    ##  (within 30 residues from both ends),
                    ##  THEN cut them
                    if (($dirj eq "for" && $posj1 <= 30) ||
                        ($dirj eq "rev" && $posj2 >= $qlenj - 30 + 1)) {
                        foreach $m ($k..$enddata) {
                            $score{$j,$m} = -3;
                        }
                    }
                }

                if ($posext > 0) {
                    if ($pepov / $qlenj >= 0.7 ||
                        ($jlen / $qlenj >= 0.7 && $pepov / $klen >= 0.8) ||
                        ($klen / $qlenk >= 0.7 && $pepov / $jlen >= 0.8)) {           ## SINGLE EXON GENE
                        $score{$j,$k} = -5;
#S                    } elsif ($pepov / $klen == 1.0 || $pepov / $jlen == 1.0) {        ## ONE INCLUDED IN ANOTHER
#S                        $score{$j,$k} = -7;
                    } elsif ($gnmdist > $maxintron) {                                 ## MAXINTRON
                        $score{$j,$k} = -8;
                    } else {                                                          ## EXTENSION = [SCORE - GAP]
                        $score{$j,$k} = $d2score{abs($posdif)} if (!$score{$j,$k});
                        foreach $m ($j+1..$k-1) {
                            $evalm = (split(/\s+/, $sortdata[$m]))[3];
                            $evalm = 1e-200 if ($evalm < 1e-200);
                            $score{$j,$k} -= -log($evalm)/log(10);
                        }
                    }
                } else {                                                          ## NO EXTENSION IN PEP POS
                    $score{$j,$k} = -10;
                }
            } else {                                                          ## OPPOSITE DIRECTION
                $score{$j,$k} = -20;
            }
            printf "  %3d %4d %4d %7.2f    ", $k, $posdif, $posext, $score{$j,$k} if ($testmode);    ##  TEST  ##
        }
        print "\n" if ($testmode);     ##  TEST  ##
    }
    print "---------------------------------------------------------\n" if ($testmode);    ##  TEST  ##

    undef(@falive);
    undef(@linescore);
    foreach $j (0..$#sortdata) {
        $falive[$j] = 1;
        $tmpe = (split(/\s+/, $sortdata[$j]))[3];
        $tmpe = 1e-200 if ($tmpe == 0);
        $tmpscore = -log($tmpe)/log(10);
        $linescore[$j] = $tmpscore;
    }

    undef(%frags);
    undef(%nfrag2score);
    $nfrag = 0;
    $tmpfrag = 0;
    push(@{$frags{$nfrag}}, $tmpfrag);
    $nfrag2score{$nfrag} += $linescore[0];
    undef(@skipped);
    while ($tmpfrag <= $#sortdata) {
        $enddata = $tmpfrag + $lookup;
        $ensdata = $#sortdata if ($enddata > $#sortdata);
        $bestscore = 0;
        $bestfrag = 0;
        foreach $k ($tmpfrag + 1..$enddata) {
            if ($score{$tmpfrag, $k} > $bestscore) {
                $bestscore = $score{$tmpfrag, $k};
                $bestfrag = $k;
            }
        }
        if ($bestscore > 0) {
            push(@{$frags{$nfrag}}, $bestfrag);
            $nfrag2score{$nfrag} += $linescore[$bestfrag];
            foreach $k ($tmpfrag + 1..$bestfrag - 1) {
                push(@skipped, $sortdata[$k]);
            }
            $tmpfrag = $bestfrag;
        } else {
            ++$tmpfrag;
            if ($tmpfrag <= $#sortdata) {
                ++$nfrag;
                push(@{$frags{$nfrag}}, $tmpfrag);
                $nfrag2score{$nfrag} += $linescore[$tmpfrag];
            }
        }
    }

    sub hashvaluesort { $nfrag2score{$b} <=> $nfrag2score{$a}; }

#    print "------------------\n";
#    foreach $j (0..$nfrag) {
#        printf "$j  %8.2f  @{$frags{$j}}\n", $nfrag2score{$j};
#    }
#    print "------------------\n";

#    foreach $j (sort hashvaluesort (keys(%nfrag2score))) {
#        printf "$j  %8.2f  @{$frags{$j}}\n", $nfrag2score{$j};
#    }

    #------------#
    #   OUTPUT   #
    #------------#

    if ($geneonly eq "off") {
        if ($gff eq "off") {
            print "#DELETED#\n";
            foreach $j (0..$#deleted) {
                print "$deleted[$j]\n";
            }
 
            print "#SKIPPED#\n";
            foreach $j (0..$#skipped) {
                print "$skipped[$j]\n";
            }
        } else {                                     ###    GFF    ##
            $ndel = 0;
            foreach $j (0..$#deleted) {
                ++$ndel;
                @fields = split(/\s+/, $deleted[$j]);
                $dir = "+";
                if ($fields[14] =~ /^[\-\+][123]$/) {    ##  tblastn  ##
                    $dir = "-" if ($fields[14] < 0);
                } elsif ($fields[14] eq "Plus") {        ##  blastn  ##
                    $dir = "-" if ($fields[15] eq "Minus");
                }
                print "$fields[1]\tblast2gene\tHSP\t$fields[8]\t$fields[10]\t.\t$dir\t.\tdeleted_$ndel\n";
            }

            $nskp = 0;
            foreach $j (0..$#skipped) {
                ++$nskp;
                @fields = split(/\s+/, $skipped[$j]);
                $dir = "+";
                if ($fields[14] =~ /^[\-\+][123]$/) {    ##  tblastn  ##
                    $dir = "-" if ($fields[14] < 0);
                } elsif ($fields[14] eq "Plus") {        ##  blastn  ##
                    $dir = "-" if ($fields[15] eq "Minus");
                }
                print "$fields[1]\tblast2gene\tHSP\t$fields[8]\t$fields[10]\t.\t$dir\t.\tskipped_$nskp\n";
            }
        }
    }

    $ngene = 0;
    $nfragment = 0;
#    foreach $j (sort hashvaluesort (keys(%nfrag2score))) {
    foreach $j (0..$nfrag) {
        last unless (@sortdata);  # added by Dr. Gabor Toth
        @gline = @{$frags{$j}};

        @dat = split(/\s+/, $sortdata[$gline[0]]);
        $tmpqid = $dat[0];
        $tmpsid = $dat[1];
        $qlen = $dat[7];

        if ($dat[14] =~ /^[\-\+][123]$/) {
            if ($dat[14] > 0) {
                $genedir = "for";
                $gpos1 = $dat[8];
                $gpos2 = $dat[10];
            } else {
                $genedir = "rev";
                $gpos1 = $dat[10];
                $gpos2 = $dat[8];
            }
        } elsif ($dat[14] eq "Plus") {
            if ($dat[15] eq "Plus") {
                $genedir = "for";
                $gpos1 = $dat[8];
                $gpos2 = $dat[10];
            } else {
                $genedir = "rev";
                $gpos1 = $dat[10];
                $gpos2 = $dat[8];
            }
        } else {
            $genedir = "for";
            $gpos1 = $dat[8];
            $gpos2 = $dat[10];
        }

        $hitseq = '0' x $qlen;
        $totalilen = 0;
        $totmatch = 0;
        foreach $k (0..$#gline) {
            @dat = split(/\s+/, $sortdata[$gline[$k]]);
            $pos1 = $dat[4];
            $pos2 = $dat[6];
            foreach $l ($pos1 - 1..$pos2 - 1) {
                substr($hitseq, $l, 1) = "1";
            }

            if ($dat[14] =~ /^[\-\+][123]$/) {
                if ($dat[14] > 0) {
                    $gpos1 = $dat[8] if ($dat[8] < $gpos1);
                    $gpos2 = $dat[10] if ($dat[10] > $gpos2);
                } else {
                    $gpos1 = $dat[10] if ($dat[10] < $gpos1);
                    $gpos2 = $dat[8] if ($dat[8] > $gpos2);
                }
            } elsif ($dat[14] eq "Plus") {
                if ($dat[15] eq "Plus") {
                    $gpos1 = $dat[8] if ($dat[8] < $gpos1);
                    $gpos2 = $dat[10] if ($dat[10] > $gpos2);
                } else {
                    $gpos1 = $dat[10] if ($dat[10] < $gpos1);
                    $gpos2 = $dat[8] if ($dat[8] > $gpos2);
                }
            } else {
                $gpos1 = $dat[8] if ($dat[8] < $gpos1);
                $gpos2 = $dat[10] if ($dat[10] > $gpos2);
            }

            $totalilen += $dat[12];
            $totmatch += $dat[12] * $dat[13] / 100;
        }
        $hitseq =~ s/0//g;
        $coverage = length($hitseq)/$qlen;

        $identity = $totmatch / $totalilen * 100;

        if ($genedir eq "for") {
            $outrange = "$gpos1..$gpos2";
        } else {
            $outrange = "~($gpos1..$gpos2)";
        }

        if ($coverage >= $mincov) {
            ++$ngene;
            if ($gff eq "off") {
                if ($geneonly eq "off") {
                    if ($noscore eq "off") {
                        printf "%-14s    coverage= %5.3f    score= %8.2f    identity= %5.1f    $outrange\n",
                         "#GENE_${ngene}#", $coverage, $nfrag2score{$j}, $identity;
                    } else {
                        printf "%-14s    coverage= %5.3f    identity= %5.1f    $outrange\n",
                         "#GENE_${ngene}#", $coverage, $identity;
                    }
                } elsif ($geneonly eq "on") {
                    if ($noscore eq "off") {
                        printf "%-14s    coverage= %5.3f    score= %8.2f    identity= %5.1f    $outrange\n",
                         "GENE_${ngene}", $coverage, $nfrag2score{$j}, $identity;
                    } else {
                        printf "%-14s    coverage= %5.3f    identity= %5.1f    $outrange\n",
                         "GENE_${ngene}", $coverage, $identity;
                    }
                }
            } else {                                      ####    GFF    ####
                foreach $k (0..$#gline) {
                    @fields = split(/\s+/, $sortdata[$gline[$k]]);
                    $dir = "+";
                    if ($fields[14] =~ /^[\-\+][123]$/) {    ##  tblastn  ##
                        $dir = "-" if ($fields[14] < 0);
                    } elsif ($fields[14] eq "Plus") {        ##  blastn  ##
                        $dir = "-" if ($fields[15] eq "Minus");
                    }
                    print "$fields[1]\tblast2gene\tHSP\t$fields[8]\t$fields[10]\t.\t$dir\t.\tGENE_$ngene\n";
                }
            }
        } else {
            ++$nfragment;
            if ($gff eq "off") {
                if ($geneonly eq "off") {
                    if ($noscore eq "off") {
                        printf "%-14s    coverage= %5.3f    score= %8.2f    identity= %5.1f    $outrange\n",
                         "#FRAGMENT_${nfragment}#", $coverage, $nfrag2score{$j}, $identity;
                    } else {
                        printf "%-14s    coverage= %5.3f    identity= %5.1f    $outrange\n",
                         "#FRAGMENT_${nfragment}#", $coverage, $identity;
                    }
                }
            } else {                                      ####    GFF    ####
                foreach $k (0..$#gline) {
                    @fields = split(/\s+/, $sortdata[$gline[$k]]);
                    $dir = "+";
                    if ($fields[14] =~ /^[\-\+][123]$/) {    ##  tblastn  ##
                        $dir = "-" if ($fields[14] < 0);
                    } elsif ($fields[14] eq "Plus") {        ##  blastn  ##
                        $dir = "-" if ($fields[15] eq "Minus");
                    }
                    print "$fields[1]\tblast2gene\tHSP\t$fields[8]\t$fields[10]\t.\t$dir\t.\tFRAGMENT_$nfragment\n";
                }
            }
        }
        
        if ($gff eq "off" && $geneonly eq "off") {
            foreach $k (0..$#gline) {
                print "$sortdata[$gline[$k]]\n";
            }
        }
    }

    if ($gff eq "off" && ($geneonly eq "off" || $pagen eq "all")) {
        print "##  $tmppair / $totpair        $tmpqid    $tmpsid\n";
    }

  }
}

#-----------------------------------------------------------------------

sub overlen {
    local($a, $b, $c, $d) = @_;
    local($olen);

    #  frag_1    |------|
    #            a      b
    #  frag_2    |--------|
    #            c        d

    if ($a > $b || $c > $d) {
        return("range_error");
    } elsif ($a == $c && $b == $d) {
        #       |-----|
        #       |-----|
#        return("O_121212");
        $olen = $b - $a + 1;
        return($olen);
    } elsif ($b > $c && $b < $d && $a < $c) {
        #       |----|
        #          |----|
#        return("O_112122");
        $olen = $b - $c + 1;
        return($olen);
    } elsif ($a < $c && $b == $c && $d > $b) {
        #       |----|
        #            |----|
#        return("O_111222");
        $olen = 1;
        return($olen);
    } elsif ($c < $a && $d > $a && $d < $b) {
        #          |----|
        #       |----|
#        return("O_221211");
        $olen = $d - $a + 1;
        return($olen);
    } elsif ($c < $a && $a == $d && $b > $d) {
        #            |----|
        #       |----|
#        return("O_222111");
        $olen = 1;
        return($olen);
    } elsif ($a >= $c && $b <= $d) {
        #         |---|
        #       |--------|
#        return("O_221122");
        $olen = $b - $a + 1;
        return($olen);
    } elsif ($c >= $a && $d <= $b) {
        #       |--------|
        #         |---|
#        return("O_112211");
        $olen = $d - $c + 1;
        return($olen);
    } elsif ($b < $c) {
        #       |---|
        #              |---|
#        return("N_110022");
        $olen = 0;
        return($olen);
    } elsif ($a > $d) {
        #              |---|
        #       |---|
#        return("N_220011");
        $olen = 0;
        return($olen);
    } else {
        return("U");
    }
}
