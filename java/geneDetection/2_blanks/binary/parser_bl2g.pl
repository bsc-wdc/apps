#/usr/bin/perl

my $report = shift;

my $infile= shift;
my $infile2=shift;

#print ">$infile";

open IN, ">$infile" || return "";
open IN2, ">$infile2" || return "";

my $num = 1;
my $db = "seq.fasta";

my $strand;
my $start;
my $end;
my $protid;
my $cov;
my $ident;
my $type;

open(F,$report);
                my $first = 1;
                while(<F>){

                        if ($first && /^([^#]\S+)\s+.*/){
                                $protid = $1;
                                if ($protid =~ /\S+\|(\S+)/){
                                        $protid = $1;
                                }
                                $first = 0;

                                print IN "$num\t$db\t";
                                if ($strand eq "Reverse"){
                                        print IN "~($start..$end)";
                                } else {
                                        print IN "$start..$end";
                                }
                                print IN  "\n";


                                print IN2 "$num $protid $cov $ident $type $start $end $strand $db\n";


                                $num = $num+1;
                        }

                     if (/#(\S+)_.+#\s+coverage=\s+(\S+).+identity=\s+(\S+)\s+.*\((\S+)\.\.(\S+)\)/){
                                $first = 1;
                                $strand = "Reverse";
                                if ($4 eq "-"){
                                        $start = 1;
                                } else {
                                        $start = $4;
                                }
                                $end = $5;
                                $cov = $2;
                                $ident = $3;
                                if ($1 eq "GENE"){
                                        $type = "GENE";
                                } else {
                                        $type = "FRAGMENT";
                                }
                       } elsif (/#(\S+)_.+#\s+coverage=\s+(\S+).+identity=\s+(\S+)\s+(\S+)\.\.(\S+)/){
                                $first = 1;
                                $strand = "Forward";
                                $start = $4;
                                $end = $5;
                                $cov = $2;
                                $ident = $3;
                                if ($1 eq "GENE"){
                                        $type = "GENE";
                                } else {
                                        $type = "FRAGMENT";
                                }

                        }
                }

                close(F);



        close IN;
        close IN2;

