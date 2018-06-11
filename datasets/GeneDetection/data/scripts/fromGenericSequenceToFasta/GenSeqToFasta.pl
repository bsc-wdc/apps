#!/usr/bin/perl

my $infile = shift;
my $id = shift;
my $ns = shift;

open(F,"$infile");
my $str = "<FASTA id='$id' namespace='$ns'>";
$str .= "<String articleName='content'><![CDATA[";
while(<F>){
	$str .= $_;
}
$str .= "]]></String></FASTA>";

print $str;
