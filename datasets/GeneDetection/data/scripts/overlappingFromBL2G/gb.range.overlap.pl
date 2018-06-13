#!/usr/bin/perl
#
#   Usage:  gb.range.overlap.pl  infile1  infile2  [-nodir]
#
#
#    infile:
#       ID    chr    range
#
#    If infile(1 and 2) contains duplicated IDs,
#    the program will print warning message to STDERR
#    and only take the first ID.
#
#
#                                          18/05/2004    Mikita Suyama
#    v3:
#      - output is the same order in INFILE1
#      - get pos1 & pos2 when read the files (%id{ij}2pos1, %id{ij}2pos2);
#      - group infile2 by chr && direction
#                                          14/06/2004
#
#    v4:
#      - group infile2 by chr (not chr && direction)
#      - '-nodir' option: don't take direction into account
#                                          18/06/2004

$| = 1;

if ($#ARGV < 1) {
    print STDERR "\n";
    print STDERR "Usage:  gb.range.overlap.pl  infile1  infile2  [-nodir]\n";
    print STDERR "\n";
    print STDERR "    infile:\n";
    print STDERR "        ID    chr    range\n";
    print STDERR "\n";
    exit;
} else {
    $dirop = "on";
    foreach $i (0..$#ARGV) {
        if ($ARGV[$i] eq "-nodir") {
            $dirop = "off";
        } elsif (!$infile1) {
            $infile1 = $ARGV[$i];
        } elsif (!$infile2) {
            $infile2 = $ARGV[$i];
        }
    }
}


#--------------------------------------------------------#
#   READ INFILE1
#
#     if there are duplicated IDs,
#     print warning and take the first one.
#--------------------------------------------------------#

open(INFILE1, "< $infile1") || die "Can't open $infile1";
undef(@idi);
undef(%idi2chr);
undef(%idi2range);
undef(%idi2pos1);
undef(%idi2pos2);
$longestid1 = 0;
while (<INFILE1>) {
    chomp;
    ($tmpid, $tmpchr, $tmprange) = split(/\s+/, $_);
    $longestid1 = length($tmpid) if (length($tmpid) > $longestid1);
    if (!$idi2range{$tmpid}) {
        $idi2chr{$tmpid} = $tmpchr;
        push(@idi, $tmpid);
        $idi2range{$tmpid} = $tmprange;
        $modrange = $tmprange;
        $modrange =~ s/~\(//;
        $modrange =~ s/join\(//;
        $modrange =~ s/\)//g;
        @ranges = split(/\.\./, $modrange);
        $idi2pos1{$tmpid} = $ranges[0];
        $idi2pos2{$tmpid} = $ranges[$#ranges];
    } else {
        print STDERR "\nWARNING: duplicated ID\n";
        print STDERR "    infile1 = $infile1\n";
        print STDERR "    dup ID = $tmpid\n";
        print STDERR "    range = $tmprange\n";
        print STDERR "  for this id the following range is used:\n";
        print STDERR "    range = $idi2range{$tmpid}\n";
        print STDERR "\n";
    }
}
close(INFILE1);


#--------------------------------------------------------#
#   READ INFILE2
#     if there are duplicated IDs,
#     print warning and take the first one.
#--------------------------------------------------------#

open(INFILE2, "< $infile2") || die "Can't open $infile2";
undef(%chr2idjarr);
undef(%idj2range);
undef(%idj2pos1);
undef(%idj2pos2);
$longestid2 = 0;
while (<INFILE2>) {
    chomp;
    ($tmpid, $tmpchr, $tmprange) = split(/\s+/, $_);
    $longestid2 = length($tmpid) if (length($tmpid) > $longestid2);
    if (!$idj2range{$tmpid}) {
        push(@{$chr2idjarr{$tmpchr}}, $tmpid);
        $idj2range{$tmpid} = $tmprange;
        $modrange = $tmprange;
        $modrange =~ s/~\(//;
        $modrange =~ s/join\(//;
        $modrange =~ s/\)//g;
        @ranges = split(/\.\./, $modrange);
        $idj2pos1{$tmpid} = $ranges[0];
        $idj2pos2{$tmpid} = $ranges[$#ranges];
    } else {
        print STDERR "\nWARNING: duplicated ID\n";
        print STDERR "    infile2 = $infile2\n";
        print STDERR "    dup ID = $tmpid\n";
        print STDERR "    range = $tmprange\n";
        print STDERR "  for this id the following range is used:\n";
        print STDERR "    range = $idj2range{$tmpid}\n";
        print STDERR "\n";
    }
}
close(INFILE2);


#-----------------------------------------#
#   SORT J (for each chr, sort by pos1)
#
#
#      |-------|
#       |----|
#         |---------------|
#           |-|
#            |-------|
#                |-----|
#                  |---------|
#                    |---|
#
#-----------------------------------------#

foreach (keys %chr2idjarr) {
    @tmpidj = @{$chr2idjarr{$_}};
    local(@datakeys);
    foreach (@tmpidj) {
        push(@datakeys, $idj2pos1{$_});
    }
    sub bydatakeys { $datakeys[$a] <=> $datakeys[$b]; }
    @sortdata = @tmpidj[sort bydatakeys $[..$#tmpidj];
    @{$chr2idjarr{$_}} = @sortdata;
}


#--------------#
#   COMPARE 
#--------------#

foreach $i (0..$#idi) {
    @subidj = @{$chr2idjarr{$idi2chr{$idi[$i]}}};
    
    $hitfound = 0;
    foreach $j (0..$#subidj) {


        #   i     |---|
        #             ^
        #   j  |-------|
        #       |----|
        #         |---------------|
        #           |-|
        #            |-------|
        #                |-----|        <-- last
        #                ^ |---------|
        #                    |---|
        #

        if ($idj2pos1{$subidj[$j]} <= $idi2pos2{$idi[$i]}) {
            if ($idj2pos2{$subidj[$j]} >= $idi2pos1{$idi[$i]}) {

                %gbovout = & gboverlap($idi2range{$idi[$i]}, $idj2range{$subidj[$j]});
                if ($dirop eq "on") {
                    if ($gbovout{'direction'} eq "same" && $gbovout{'cds_overlap'}) {
                        printf "%-${longestid1}s    %-${longestid2}s  %5d / %5d  %5.3f    %5d / %5d  %5.3f\n",
                         $idi[$i], $subidj[$j],
                         $gbovout{'cds_overlap'}, $gbovout{'gb1_cds_len'}, $gbovout{'cds_overlap'} / $gbovout{'gb1_cds_len'},
                         $gbovout{'cds_overlap'}, $gbovout{'gb2_cds_len'}, $gbovout{'cds_overlap'} / $gbovout{'gb2_cds_len'};
                        $hitfound = 1;
                    }
                } else {
                    if ($gbovout{'cds_overlap'}) {
                        printf "%-${longestid1}s    %-${longestid2}s  %5d / %5d  %5.3f    %5d / %5d  %5.3f\n",
                         $idi[$i], $subidj[$j],
                         $gbovout{'cds_overlap'}, $gbovout{'gb1_cds_len'}, $gbovout{'cds_overlap'} / $gbovout{'gb1_cds_len'},
                         $gbovout{'cds_overlap'}, $gbovout{'gb2_cds_len'}, $gbovout{'cds_overlap'} / $gbovout{'gb2_cds_len'};
                        $hitfound = 1;
                    }
                }
            }
        } else {
            last;
        }
    }

    printf "%-${longestid1}s    No_hits_found\n", $idi[$i] if ($hitfound == 0);

    print "//\n";
        
}


#---------------------------

sub gboverlap {
    #  input  gboverlap($range1, $range2);
    #  output hash
    #           'direction'        direction ('same' or 'opp');
    #           'cds_overlap'      CDS overlap (in total length);
    #           'gb1_cds_len'
    #           'gb2_cds_len'
    #           'ini_end_overlap'  ini-end range overlap (in total length);
    #                              CDS may not overlap
    #           'gb1_ini_end_len'
    #           'gb2_ini_end_len'
    #
    #           'ov_coord'
    #           'in1_coord'
    #           'in2_coord'
    #

    local($range1, $range2) = @_;
    local($dir1, $dir2, %retval, @rangearr1, @rangearr2, $i, $j);
    local($posi1, $posi2, $posj1, $posj2);
    local($ovgb, $in1gb, $in2gb);
    local(@ovranges);
    local(@in1ranges);
    local(@in2ranges);
    undef(%retval);

    local($minpos, $maxpos);
    local(@negranges);


    #--------------------------#
    #  $retval{'direction'}
    #--------------------------#

    if ($range1 =~ s/^~\(//) {
        $dir1 = "rev";
        $range1 =~ s/\)$//;
    } else {
        $dir1 = "for";
    }
    if ($range2 =~ s/^~\(//) {
        $dir2 = "rev";
        $range2 =~ s/\)$//;
    } else {
        $dir2 = "for";
    }
    if ($dir1 eq $dir2) {
        $retval{'direction'} = "same";
    } else {
        $retval{'direction'} = "opp";
    }


    #-----------#

    $range1 =~ s/join\(//;
    $range1 =~ s/\)//;
    @rangearr1 = split(/,/, $range1);

    $range2 =~ s/join\(//;
    $range2 =~ s/\)//;
    @rangearr2 = split(/,/, $range2);


    #--------------------------------#
    #  $retval{'ini_end_overlap'}
    #--------------------------------#

    $posi1 = (split(/\.\./, $rangearr1[0]))[0];
    $posi2 = (split(/\.\./, $rangearr1[$#rangearr1]))[1];

    $posj1 = (split(/\.\./, $rangearr2[0]))[0];
    $posj2 = (split(/\.\./, $rangearr2[$#rangearr2]))[1];

    %ovlenout = & overlen($posi1, $posi2, $posj1, $posj2);
    $retval{'ini_end_overlap'} = $ovlenout{'length'};
    $retval{'gb1_ini_end_len'} = $posi2 - $posi1 + 1;
    $retval{'gb2_ini_end_len'} = $posj2 - $posj1 + 1;

    $minpos = $posi1;
    $minpos = $posj1 if ($posj1 < $minpos);

    $maxpos = $posi2;
    $maxpos = $posj2 if ($posj2 > $maxpos);

    #----------------------------#
    #  $retval{'cds_overlap'}
    #----------------------------#

    $cdsoverlap = 0;

    if ($retval{'ini_end_overlap'} <= 0) {

        $retval{'cds_overlap'} = 0;

        $retval{'in1_coord'} = "join(" if ($#rangearr1 >= 1);
        $retval{'in1_coord'} .= $range1;
        $retval{'in1_coord'} .= ")" if ($#rangearr1 >= 1);

        $retval{'in2_coord'} = "join(" if ($#rangearr2 >= 1);
        $retval{'in2_coord'} .= $range2;
        $retval{'in2_coord'} .= ")" if ($#rangearr2 >= 1);

    } else {
        foreach $i (0..$#rangearr1) {
            ($posi1, $posi2) = split(/\.\./, $rangearr1[$i]);
            foreach $j (0..$#rangearr2) {
                ($posj1, $posj2) = split(/\.\./, $rangearr2[$j]);
                %ovlenout = & overlen($posi1, $posi2, $posj1, $posj2);
                $tmpov = $ovlenout{'length'};
                if ($tmpov >= 1) {
                    push(@ovranges, "$ovlenout{'pos1'}..$ovlenout{'pos2'}");
                }
                $cdsoverlap += $tmpov if ($tmpov >= 0);
            }
        }
        $retval{'cds_overlap'} = $cdsoverlap;

        $retval{'ov_coord'} = "join(" if ($#ovranges >= 1);
        foreach $i (0..$#ovranges) {
            $retval{'ov_coord'} .= $ovranges[$i];
            $retval{'ov_coord'} .= "," if ($i != $#ovranges);

            ($posi1, $posi2) = split(/\.\./, $ovranges[$i]);
            if ($i == 0) {
                if ($minpos < $posi1) {
                    --$posi1;
                    push(@negranges, "$minpos..$posi1");
                }
                if ($i != $#ovranges) {
                    ($posj1, $posj2) = split(/\.\./, $ovranges[$i + 1]);
                    ++$posi2;
                    --$posj1;
                    push(@negranges, "$posi2..$posj1");
                }
            } elsif ($i == $#ovranges) {
                if ($maxpos > $posi2) {
                    ++$posi2;
                    push(@negranges, "$posi2..$maxpos");
                }
            } else {
                ($posi1, $posi2) = split(/\.\./, $ovranges[$i - 1]);
                ($posj1, $posj2) = split(/\.\./, $ovranges[$i]);
                ++$posi2;
                --$posj1;
                push(@negranges, "$posi2..$posj1");
                ($posi1, $posi2) = split(/\.\./, $ovranges[$i]);
                ($posj1, $posj2) = split(/\.\./, $ovranges[$i + 1]);
                ++$posi2;
                --$posj1;
                push(@negranges, "$posi2..$posj1");
            }
        }
        $retval{'ov_coord'} .= ")" if ($#ovranges >= 1);

        #-----------------------------------#
        #  in1_coord, in2_coord
        #-----------------------------------#

        foreach $i (0..$#rangearr1) {
            ($posi1, $posi2) = split(/\.\./, $rangearr1[$i]);
            foreach $j (0..$#negranges) {
                ($posj1, $posj2) = split(/\.\./, $negranges[$j]);
                %ovlenout = & overlen($posi1, $posi2, $posj1, $posj2);
                if ($ovlenout{'length'} >= 1) {
                    push(@in1ranges, "$ovlenout{'pos1'}..$ovlenout{'pos2'}");
                }
            }
        }
        $retval{'in1_coord'} = "join(" if ($#in1ranges >= 1);
        foreach $i (0..$#in1ranges) {
            $retval{'in1_coord'} .= $in1ranges[$i];
            $retval{'in1_coord'} .= "," if ($i != $#in1ranges);
        }
        $retval{'in1_coord'} .= ")" if ($#in1ranges >= 1);


        foreach $i (0..$#rangearr2) {
            ($posi1, $posi2) = split(/\.\./, $rangearr2[$i]);
            foreach $j (0..$#negranges) {
                ($posj1, $posj2) = split(/\.\./, $negranges[$j]);
                %ovlenout = & overlen($posi1, $posi2, $posj1, $posj2);
                if ($ovlenout{'length'} >= 1) {
                    push(@in2ranges, "$ovlenout{'pos1'}..$ovlenout{'pos2'}");
                }
            }
        }
        $retval{'in2_coord'} = "join(" if ($#in2ranges >= 1);
        foreach $i (0..$#in2ranges) {
            $retval{'in2_coord'} .= $in2ranges[$i];
            $retval{'in2_coord'} .= "," if ($i != $#in2ranges);
        }
        $retval{'in2_coord'} .= ")" if ($#in2ranges >= 1);

    }


    foreach $i (0..$#rangearr1) {
        ($posi1, $posi2) = split(/\.\./, $rangearr1[$i]);
        $retval{'gb1_cds_len'} += $posi2 - $posi1 + 1;
    }
    foreach $j (0..$#rangearr2) {
        ($posj1, $posj2) = split(/\.\./, $rangearr2[$j]);
        $retval{'gb2_cds_len'} += $posj2 - $posj1 + 1;
    }


    #---------------------------------

    %retval;
}


#-----------------------------------------------------------------------

sub overlen {
    local($a, $b, $c, $d) = @_;
    local(%olres);

    #  input   overlen($f1p1, $f1p2, $f2p1, $f2p2);
    #  output  hash
    #            'length'
    #            'pos1'
    #            'pos2'


    #  frag_1    |------|
    #            a      b
    #  frag_2    |--------|
    #            c        d

    #-------------------------------#
    #   Are they proper direction?
    #-------------------------------#

    if ($a > $b || $c > $d) {
        print STDERR "SUB:overlen  range_error, must be a <= b and c <= d\n";
        print STDERR "             a-b  $a  $b\n";
        print STDERR "             c-d  $c  $d\n$_\n";
        exit;
    }


    #----------------------------------#
    #   return if they don't overlap
    #----------------------------------#

    if ($b < $c) {
        #       a-----b
        #                 c----d
        return;
    } elsif ($d < $a) {
        #                 a----b
        #       c-----d
        return;
    }


    #--------------------------------------------#
    #   calc length of overlap if they overlap
    #--------------------------------------------#

    if ($b >= $c && $b <= $d) {
        if ($a <= $c) {
            #       a-----b
            #          c------d
            $olres{'length'} = $b - $c + 1;
            $olres{'pos1'} = $c;
            $olres{'pos2'} = $b;
            return(%olres);
        } else {
            #       a-----b
            #     c-----------d
            $olres{'length'} = $b - $a + 1;
            $olres{'pos1'} = $a;
            $olres{'pos2'} = $b;
            return(%olres);
        }
    } elsif ($d >= $a && $d <= $b) {
        if ($c <= $a) {
            #          a------b
            #       c-----d
            $olres{'length'} = $d - $a + 1;
            $olres{'pos1'} = $a;
            $olres{'pos2'} = $d;
            return(%olres);
        } else {
            #     a-----------b
            #       c-----d
            $olres{'length'} = $d - $c + 1;
            $olres{'pos1'} = $c;
            $olres{'pos2'} = $d;
            return(%olres);
        }
    }
}
