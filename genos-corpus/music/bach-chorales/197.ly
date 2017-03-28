\version "2.10.33"

\header {
  title = "197 - Christ ist erstanden"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \major
}


soprano = \relative c'' {
  \partial 4 a4 
  g4 a8 b8 c4( d4)
  a2 a4 f8 g8 
  a8 g8 f4 e8( d8 e4)
  d2 r4 f4 
  g4 g4 d4 c4   %5
  f4( g4) a4 a8 g8 
  f4 g4 a8 g8 f4 
  e4( d4) e2 
  d4 d4 d2 
  d1   %10
  a'4 a4 g4 a8 bes8 
  c4( d4) a4 a4 
  a8. g16 f8 g8 a8 g8 f4 
  e8( d8 e4) d4 f4 
  g4 g4 d4 c4   %15
  f4 g4 a4 a4 
  a8 g8 f8. g16 a4 f4 
  e4 d4 e2 
  d4 d4 d2
  d1   %20
  \partial 4
  a'4 
  f4 c'4 a4 c8 bes8 
  a4 g4 f4 a8 g8 
  f4 e4 d4 f4 
  g4 g4 d4 c4 
  f4( g4) a4  a8 g8 %25
  f4 g4 a4 f4 
  e4( d4) e2 
  d4 d4 d2
  d1
}

alto = \relative c' {
  \partial 4 f4~
  f8 e8 e8 gis8 a4. g8~
  g8 f16 e16 f4 e4 d4
  c8 cis8 d4 cis8 d4 cis8
  a2 r4 d4~
  d8 c16 b16 c4~ c8 b4 a16 bes16  %5
  c8 d8 e4 f4 e4~
  e8 d4 cis8 d4. c8
  bes8 a4 b8 cis2 
  d4 c4 c4 bes4~
  bes4 a8 g8 a2   %10
  f'4 e8 d8 g8 bes8 a8 g8
  f2 f4 f4
  e4 d8 e8 f8 cis8 d4~
  d4 cis4 d4 c4
  c8 b8 c4~ c8 bes4 a16 g16  %15
  c4 c4 c4 c8 cis8
  d4 a4 d4 d4~ 
  d8 c4 b8 c2 
  c8 bes8 a4 g4. bes8
  a1   %20
  \partial 4 e'4
  d4 c4 c4 c8 d8
  e8 f4 e8 c4 e4~
  e8 d4 c8 c8 bes8 c4
  d4 c8 b8 a8 b8 c8 bes8
  a16 g16 a8 d8 c8 c4 f8 e8  %25
  d4. c16 bes16 a4. d16 c16
  b8 a4 gis8 cis2
  d4 c4 bes8 a8 g16 a16 bes8~
  bes8 a8 bes8 g8 a2 
}

tenor = \relative c' {
  \partial 4 d8 c8 
  b4 c8 d8 e8 f16 e16 d4 
  d2 a4 a4 
  a4 a4 bes8 f8 g8 a16 g16 
  f2 r4 a4 
  e8 f8 g4~ g16 d16 e8~ e8 f16 g16   %5
  a8 f8 bes4 c4 a4 
  a4 g8 bes8 a4~ a16 g16 a8 
  g16 f16 g8 f16 e16 f8 e2 
  a8 g8 a8 fis8 g8 d8 g4~ 
  g4 fis8 e8 fis2   %10
  d'4 c8 b8 c4 c4 
  c4. bes8 c4 c8 d8 
  e8 a,4 g8 f8 g8 a4 
  a4. g8 f4 a4 
  g8 f8 g4 g8. f16 e4   %15
  a4 g4 f4 a8 g8 
  f8 e8 d8 e8 f8 g8 a4 
  a8 e8 f4 g2 
  a8 g4 fis8 g8 d8 g4~ 
  g4 fis8 e8 fis2   %20
  \partial 4 a4~
  a8 bes8 a8 g8 f4 f4
  c'4. bes8 a4 a4
  a4 g8 a8 d,4 a'8 f8
  d16 e16 f8~ f e8 fis8 gis8 a8 g8
  f16 e16 f8~ f e8 f4 c'4~  %25
  c8 bes16 a16 g4. f16 e16 f8 bes16 a16
  gis8 a8 d8 d,8 a'2
  f8 g8 a4. g16 fis16 
  g4~ g8 fis8 g8 e8 fis2
}

baixo = \relative c {
  \partial 4 d4 
  e8. d16 c8 b8 a4 bes8. c16 
  d2 cis4 d8 e8 
  f4. f,8 g8 bes8 a4 
  d2 r4 d,4 
  e4. f8 g8 gis8 a4~   %5
  a8 bes8 a8 g8 f4 cis'4 
  d4 e4 f8 e8 d4~ 
  d8 cis8 d4 a2 
  fis8 e8 fis8 d8 g8 a8 bes8 g8 
  d'2 d,2   %10
  d'8 e8 f4. e8 f8 g8 
  a8 f8 bes4 f4 f4 
  cis4 d4. e8 f8 g8 
  a4 a,4 bes4 a4 
  e8 d8 e8 f8 g4 a4~   %15
  a8 g8 f8 e8 f4 f'8 e8 
  d2~ d8 e8 f8 g8 
  a8 a,8 d4 c2 
  fis,8 g8 c4. bes16 a16 bes8 g8 
  d'2 d,2   %20
  \partial 4 cis'4
  d4 e4 f4 a,8 bes8
  c4 c,4 f4 cis'4
  d8 d,8 e8 fis8 g4 a4
  b4 c4. b8 a4~
  a8 d8 bes8 c8 f4 f,4  %25
  bes4 ees8 d8 cis4 d4~
  d8 c8 b4 a2
  bes8 a8 g8 fis8 g8 a8 bes8 g8
  d1
}

\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}