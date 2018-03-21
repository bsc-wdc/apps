#!/usr/bin/perl -s
#
#   blast.parse.pl          Parse the blast output
#
#   Usage:  blast.parse.pl  [-ali]  [-first]  [-longname]  blast.out  outfile
#
#                                        20/07/1999    Mikita Suyama
#
#              Modification:
#                26/11/1999   In the case of 'No hits found'...
#                21/03/2000   Score, Expect... for both blast and paracel
#                07/03/2001   To deal with SeqLen contains ','
#                             (this occurs in huge sequences)
#                16/03/2001   option to get 1st hit only
#                19/03/2001   output -> subroutine
#                29/03/2001   bl2seq: no ali -> 'No hits found'
#
#                11/09/2002   normalized entropy (normalized diversity)
#                             is shown.
#                             The value is a product of H(norm; for
#                             matching fragments) and H(norm; for mismatch
#                             fragments)
# 
#                04/12/2002   very long query name (that are folded in
#                             more than one line).
#                             -> join them if the "^Query=" line end
#                                with non_spc and the next line begin
#                                with non_spc
#
#                02/01/2003   -nolongname: switch for long query name
#                             take only the first line if -nologname
#                             is specified.
#
#                24/06/2003   -longname:
#                             Now the default is "nolongname"
#
#                23/07/2003   V4
#                             Query name is missing in the text output of
#                             NCBI server:
#                             If no Qid, then write 'query'

$| = 1;

if ($#ARGV != 1) {
    print STDERR "\nUsage:  blast.parse.pl  [-ali]  [-first]  [-longname]  blast.out  outfile\n\n";
    exit;
} else {
    $infile = $ARGV[0];
    $outfile = $ARGV[1];
}

#if (-e $outfile) {
#    print STDERR "\nOutput file, $outfile, already exists!\n\n";
#    exit;
#}
open(OUTFILE, "> $outfile") || die "Can't open $outfile";
open(BOUT, "< $infile") || die "Can't open $infile";


$maxidlen = 40;

undef($sbit);
undef($exp);
undef($alilen);
undef($idp);
undef($dir1);
undef($dir2);
$frame = 0;
undef($qini);
undef($qend);
undef($sini);
undef($send);
$qid = "Query";
$nali = 0;
$nhfrag = 0;

undef($queryline);
undef($sbjctline);

$getali = 0;

my $lll = 0;

while (<BOUT>) {
    chomp;
    if (/^\s*Query=/) {
print "Query=";
        $tmpqline = $_;
        if ($qini) {
          ++$nali;
          if (!$first || $nali == 1) {
            & outwrt ($_, $maxidlen, $qid, $sid, $sbit, $exp,
                      $qini, $qend, $qlen,
                      $sini, $send, $slen,
                      $alilen, $idp, $dir1, $dir2, $frame, $ali, '1', *alidat);
          }
          undef($sbit);
          undef($exp);
          undef($alilen);
          undef($idp);
          undef($dir1);
          undef($dir2);
          $frame = 0;
          undef($qini);
          undef($qend);
          undef($sini);
          undef($send);

          print "##\n" if (!$first || $ali);
        }
        $nali = 0;
        $nhfrag = 0;
        $queryline = 1;
        undef(@alidat);
    } elsif (!$lll && /^\s*Length=\s*([0-9]+).*/){
	    $qlen = $1;
print "new qlen $_\n";
            $lll = 1;
	    $queryline = 0;

            if ($tmpqline =~ /^\s*Query=\s+(\S+)/) {    # modified on 04/12/2002
                $qid = $1;
            } else {
                $qid = "Query";
            }
            if (length($qid) > $maxidlen) {
                print  STDERR "qid > maxidlen\n";
                printf STDERR "   $qid  (%3d)\n\n", length($qid);
            }

    } elsif ($queryline) {                    #### Get query sequence length
print "queryline $_ \n";
        if (/^\s*\(([0-9\,]+) letters\)/) {
print "qlen $_\n";
            $qlen = $1;
            $qlen =~ s/\,//g;                     # remove comma (,)  07/03/2001
            $queryline = 0;

            if ($tmpqline =~ /^\s*Query=\s+(\S+)/) {    # modified on 04/12/2002
                $qid = $1;
            } else {
                $qid = "Query";
            }
            if (length($qid) > $maxidlen) {
                print  STDERR "qid > maxidlen\n";
                printf STDERR "   $qid  (%3d)\n\n", length($qid);
            }
        } else {
            $tmpqline .= $_ if ($longname);
        }
    } elsif (/ No hits found /) {             #### No hits found
        $nhfrag = 1;
        printf OUTFILE "%-${maxidlen}s  ***** No hits found ******1\n", $qid;
        printf OUTFILE "##\n" if (!$first || $ali);
        undef($qini);
        undef($qend);
        undef($qid);
        $qid = "Query";
        $nali = 0;
    } elsif (/^\s*Matrix: / && $nali == 0 && !$nhfrag && !$qini) {
        printf OUTFILE "%-${maxidlen}s  ***** No hits found ******2\n", $qid;
        printf OUTFILE "##\n" if (!$first || $ali);
        undef($qini);
        undef($qend);
        undef($qid);
        $qid = "Query";
        $nali = 0;
    } elsif (/^\s*>/) {
print ">>>>>>>>>>>>>>>>>>>>>>>>>> $_\n";
        if ($qini) {
          ++$nali;
          if (!$first || $nali == 1) {
            & outwrt ($_, $maxidlen, $qid, $sid, $sbit, $exp,
                      $qini, $qend, $qlen,
                      $sini, $send, $slen,
                      $alilen, $idp, $dir1, $dir2, $frame, $ali, '2', *alidat);
          }
          undef($sbit);
          undef($exp);
          undef($alilen);
          undef($idp);
          undef($dir1);
          undef($dir2);
          $frame = 0;
          undef($qini);
          undef($qend);
          undef($sini);
          undef($send);
        }
        @dat = split(/\s+/, $_);
        $sid = $dat[0];
        $sid =~ s/^>//;

if (/>\s+(\S+)/){
	$sid = $1;
}
        if (length($sid) > $maxidlen) {
            print  STDERR "sid > maxidlen\n";
            printf STDERR "   $sid  (%3d)\n", length($sid);
#            exit;
        }
        $sbjctline = 1;
        undef(@alidat);
    } elsif ($sbjctline) {                  #### Get subject sequence length
        if (/^\s*Length\s*=\s*([,0-9]+)/) {
            $slen = $1;
            $slen =~ s/,//g;
            $sbjctline = 0;
        }
    } elsif (/^\s*Score/) {
        if ($qini) {
          ++$nali;
          if (!$first || $nali == 1) {
              & outwrt ($_, $maxidlen, $qid, $sid, $sbit, $exp,
                        $qini, $qend, $qlen,
                        $sini, $send, $slen,
                        $alilen, $idp, $dir1, $dir2, $frame, $ali, '2', *alidat);
         
          }
          undef($sbit);
          undef($exp);
          undef($alilen);
          undef($idp);
          undef($dir1);
          undef($dir2);
          $frame = 0;
          undef($qini);
          undef($qend);
          undef($sini);
          undef($send);
        }

        @dat = split(/\s+/, $_);
        foreach $k (0..$#dat) {
            if ($dat[$k] =~ /\s*^Score/) {
                $sbit = $dat[${k}+2];
            } elsif ($dat[$k] =~ /^Expect/) {
                $exp = $dat[${k}+2];
                $exp = '1'.$exp if ($exp =~ /^e/);
                $exp =~ s/,$//;
            }
        }
        undef(@alidat);
    } elsif (/^\s*Identities/) {
        @dat = split(/\s+/, $_);
        foreach $k (0..$#dat) {
            if ($dat[$k] =~ /\s*^Identities/) {
                $idp = $dat[${k}+3];
                $idp =~ s/,//;
                $idp =~ s/^\(//;
                $idp =~ s/\)//;
                $idp =~ s/\%//;
                ($dmy, $alilen) = split(/\//, $dat[${k}+2]);
            } elsif ($dat[$k] =~ /^Strand/) {
                $dir1 = $dat[${k}+2];
                $dir2 = $dat[${k}+4];
            }
        }
        $getali = 1;
        undef(@alidat);
    } elsif (/^\s*Strand/) {
        @dat = split(/\s+/, $_);
        $dir1 = $dat[3];
        $dir2 = $dat[5];
        $getali = 1;
        undef(@alidat);
    } elsif (/^\s*Frame/) {
        @dat = split(/\s+/, $_);
        $frame = $dat[3];
        $getali = 1;
        undef(@alidat);
    } elsif (/^\s*Query/) {
print "comença queryyyyyyyyyyyy qini init!!\n";
        @dat = split(/\s+/, $_);
        $qini = $dat[1] + $tmpoffset if (!$qini);
        $qend = $dat[$#dat] + $tmpoffset;
    } elsif (/^\s*Sbjct/) {
        @dat = split(/\s+/, $_);
        $sini = $dat[1] if (!$sini);
        $send = $dat[$#dat];
    }

    if ($getali) {
        ++$getali;
        push(@alidat, "    $_") if ($getali >= 3);
    }
}
close(BOUT);

if ($qini) {
  ++$nali;
  if (!$first || $nali == 1) {
      & outwrt ($_, $maxidlen, $qid, $sid, $sbit, $exp,
                $qini, $qend, $qlen,
                $sini, $send, $slen,
                $alilen, $idp, $dir1, $dir2, $frame, $ali, '1', *alidat);
  }
  print "##\n" if (!$first || $ali);
}

close(OUTFILE);

#---------------------------------------------------------------

sub outwrt {
    local($tmpline, $maxidlen, $qid, $sid, $sbit, $exp,
          $qini, $qend, $qlen,
          $sini, $send, $slen,
          $alilen, $idp, $dir1, $dir2, $frame, $ali, $outtype, *alidat) = @_;
    local(@outarr, $i, @dat, $posidx, $collen, $seq1, $seq2, $matchseq,
          $getmatch, @matchfrag, @matchlen, @mismfrag, @mismlen,
          $normh1, $normh2);
print "YESSSSSSSSSSSSS\n";
    if ($outtype eq '1') {
        ####
        # search for the last "Sbjct"
        ####
        $lsub = 0;
        foreach $m (0..$#alidat) {
            $lsub = $m if ($alidat[$m] =~ /^    Sbjct/);
        }
        undef(@alidat2);
        foreach $m (0..$lsub + 1) {
            push(@alidat2, $alidat[$m]);
        }
        @outarr = @alidat2;
    } else {
        @outarr = @alidat;
    }
    $outali = join("\n", @outarr);

    undef($seq1);
    undef($seq2);
    undef($matchseq);
    $getmatch = 0;
    foreach $i (0..$#outarr) {
print $outarr[$i] ."XX\n";
        if ($outarr[$i] =~ /(\s*Query[:]?\s*\d+\s*)([^\s]+)\s*\d+/) {
            $posidx = length($1);
            $hitlen = length($2);
            $getmatch = 1;
            $seq1 .= substr($outarr[$i], $posidx, $hitlen);
        } elsif ($getmatch) {
            $getmatch = 0;
            $matchseq .= substr($outarr[$i], $posidx, $hitlen);
        } elsif ($outarr[$i] =~ /\s*Sbjct[:]?/) {
            $seq2 .= substr($outarr[$i], $posidx, $hitlen);
        }
    }

#t    print "$seq1\n";
#t    print "$matchseq\n";
#t    print "$seq2\n";

print "XX $matchseq\n";
    @matchfrag = split(/[\+\s]+/, $matchseq);
    splice(@matchfrag, 0, 1) if (!$matchfrag[0]);
    undef(@matchlen);
    foreach $i (0..$#matchfrag) {
#t        print "matchfrag[$i] = $matchfrag[$i]\n";
        push(@matchlen, length($matchfrag[$i]));
    }
#t    print "@matchlen\n";
print "XX @matchlen\n";

    if ($#matchlen >= 0) {
        $normh1 = & nentropy (*matchlen);
    } else {
        $normh1 = "NC";
    }
#t    print "normh1 = $normh1\n";
print "XX normh1 = $normh1\n";

    @mismfrag = split(/[A-Z]+/, $matchseq);
    splice(@mismfrag, 0, 1) if (!$mismfrag[0]);
    undef(@mismlen);
    foreach $i (0..$#mismfrag) {
#t        print "mismfrag[$i] = $mismfrag[$i]\n";
        push(@mismlen, length($mismfrag[$i]));
    }
#t    print "@mismlen\n";
    if ($#mismlen >= 0) {
        $normh2 = & nentropy (*mismlen);
    } else {
        $normh2 = "NC";
    }
#t    print "normh2 = $normh2\n";
print "XX normh2 = $normh2\n";


    if ($normh1 eq "NC" || $normh2 eq "NC") {
        $normh1x2 = "0";
    } else {
        $normh1x2 = $normh1 * $normh2;
    }
#t    print "normh1x2 = $normh1x2\n";
        

    #---------------
print "qid $qid ---- sid $sid \n";
    printf OUTFILE "%-${maxidlen}s  %-${maxidlen}s  ", $qid, $sid;
    printf OUTFILE "%8.1f %8s ", $sbit, $exp;
    printf OUTFILE "%8d - %8d %6d ", $qini, $qend, $qlen;
    printf OUTFILE "%8d - %8d %8d ", $sini, $send, $slen;
    printf OUTFILE "%5d %5.1f", $alilen, $idp;
    if ($dir1 && $dir2) {
        printf OUTFILE " %5s %5s", $dir1, $dir2;
    } elsif ($frame != 0) {
        printf OUTFILE " %2s", $frame;
    }
    printf OUTFILE " %5.3f\n", $normh1x2;

    #---------------
print "normh1x2 $normh1x2 ali $ali $outali\n";
    if ($ali) {
        print "$outali\n";
    }
}

#---------------------------------------------------------------

sub nentropy {
    local(*inarr) = @_;

    local($i, $sum, $tmpent, $p1, $maxent, $nent);
    foreach $i (0..$#inarr) {
        $sum += $inarr[$i];
    }
    foreach $i (0..$#inarr) {
        $p1 = $inarr[$i] / $sum;
        $tmpent += $p1 * (log($p1)/log(2));
    }
    $tmpent = -$tmpent;
    $maxent = log($#inarr + 1) / log(2);

    if ($maxent > 0) {
        $nent = $tmpent / $maxent;
    } else {
        $nent = 0;
    }
    $nent;
}
