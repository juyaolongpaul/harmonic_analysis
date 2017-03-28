\version "2.10.33"

\header {
  title = "132 - Kyrie, Gott Vater in Ewigkeit"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \major
}


soprano = \relative c'' {
  g2 a2 
  b2 a4 c4 
  c4 b4 a4 a4 
  g2 a4 a4 
  g4 f4 e4 e4   %5
  d2 d4 a'4 
  a4 g4 a4 b4 
  c4 b4 a2 
  g2 g4 f8 g8 
  a4 g4 g4 f4   %10
  e1 
  e2 d2 
  g2 a4 b4 
  c4 c4 c2 
  c4 c4 b4 a4   %15
  g4 a4 b4 b4 
  b2 e,4 d4 
  g4 a8 b8 c4 c4 
  c2 c4 c4 
  b4 a4 g4 a4   %20
  a4 g4 f4 e4 
  d2 d4 a'4 
  a4 g4 a4 b4 
  c4 b4 a2 
  g2 g4 f8 g8   %25
  a4 g4 g4 f4 
  e1 
  b'2 c2 
  d2 r4 b4 
  d4 c4 b4 a4   %30
  g2 b4 c4 
  d4 b8 c8 d4 c4 
  b4 a4 g4 a4 
  a4 g4 f4 e4 
  d2 d4 a'4   %35
  a4 g4 a4 b4 
  c4 b4 a2 
  g2 g4 f8 g8 
  a4 g4 g4 f4 
  e1   %40
}

alto = \relative c' {
  e8 f8 g4. fis16 e16 fis4 
  g2 f4 e4 
  a4 g4. fis16 e16 fis4 
  d2 f8 e8 d8 cis8 
  d8 cis8 d4. cis16 b16 cis4   %5
  a2 d4 e4 
  f16 e16 d4 e16 d16 c8 f4 e8~ 
  e16 fis16 g4 g4 fis16 e16 fis4 
  d2 e4 d4 
  c8 d8 e4 e4 d4   %10
  c4 b8 a8 b2 
  b4 c8 b8 a4 b8 c8 
  d8 f8 e8 d8 c8 d16 e16 f4 
  e16 f16 g4 f4 e16 d16 e4 
  e16 fis16 g4 fis4 e16 d16 cis16 b16 a8   %15
  d8 c4 a'4 g8 fis16 e16 fis8 
  e2 c4. b8 
  c8 d16 e16 f16 e16 f8 e4 f4~ 
  f8 e16 d16 e4 e8 a16 gis16 a4~ 
  a8 g4 fis8 d4 f8 g8   %20
  a8 g16 f16 f8 e4 d4 cis8 
  a2 a4 d16 e16 f8 
  f4. e8~ e16 d16 e16 g16 f16 e16 fis16 a16 
  g8 d8 d8 g4 fis16 e16 fis4 
  d2 e4 f8 e8   %25
  f4. e16 d16 e4. d16 c16 
  b8 d8 c2 b4 
  g'2 g2 
  g2 r4 g4 
  g4. a4 g4 fis8   %30
  d2 g4 a4~ 
  a8 g16 fis16 g4 g4. fis8 
  b,16 c16 d8 e8 d16 c16 b4 d4 
  d8 ees8 d8 cis8 d4. cis8 
  a2 d4 e4   %35
  a,8 fis'4 e8 fis4. e16 dis16 
  e8 d4 g4 fis16 e16 fis4 
  d2 d8 e8 d8 e8 
  f8 e8 d8 g8 cis,8 a8 d4 
  cis4 c4 b2   %40
}

tenor = \relative c' {
  b4. c16 d16 e8 a,8 d4 
  d2 d4 c8 d8 
  e2~ e8 c8 a8 d16 c16 
  b2 d8 c16 bes16 a4 
  bes4 a8 d8 bes8 g8 e8 a16 g16   %5
  f2 f4 e8 a8 
  a8 d16 c16 b8 c16 b16 a8 c8 d8 c16 b16 
  a8 g16 a16 b8 c16 d16 e8 c8 a8 d8 
  b2 c4. bes8 
  a8 b8 c8 bes8 a8 bes4 a8   %10
  a4 gis8 fis8 gis2 
  g4 a8 g8 fis4 g4~ 
  g8 d'8 c8 b8 a4. g16 f16 
  g8 c16 bes16 a4 g2 
  a4 a4 b8 g8 e'8 d16 c16   %15
  b8 c16 bes16 a8 d16 c16 b8 d8~ d16 c16 b16 a16 
  g2 g4 g4 
  g4 c8 d8 g,4 a4~ 
  a8 g16 f16 g4 c8 d8 e4~ 
  e8 d8 e8 d16 c16 b4 d4~   %20
  d8 cis8 d8 a8 a4 bes8 a16 g16 
  f2 f8 g8 a8 d8 
  d8 b8 g8 c16 b16 a4. g16 fis16 
  e8 a8 b8 e4 c8 a8 d16 c16 
  b2 c4. bes8   %25
  c16 bes16 c8 d8 g,8 c8 bes8 a8 b16 a16 
  gis4 a2 gis4 
  e'2 e2 
  b2 r4 d4 
  d4 e8 d8 d4 e8 d16 c16   %30
  b2 e4 e4 
  d4 d8 c8 b4. a8~ 
  a8 g4 fis8 g4 f8 g8 
  a8 c4 bes4 b8 e,8 a16 g16 
  fis2 b4 cis4   %35
  d4. c16 b16 c8 fis,4 g16 a16 
  g8 a4 g8 e'16 d16 c4 b16 a16 
  b2 b8 bes8 a8 g8 
  f4. e16 d16 e8 f16 g16 a8 b8~ 
  b8 a16 gis16 a4. gis16 fis16 gis4   %40
}

baixo = \relative c {
  e4. d8 c4 d4 
  g,2 d'4 a8 b8 
  c8 d8 e8 d8 c8 a8 d4 
  g,2 d'8 e8 f4~ 
  f8 e8 f8 bes8 g8 e8 a8 a,8   %5
  d2 d4. cis8 
  d8 f4 e8 f8 a4 gis8 
  a8 e4 d8 c8 a8 d4 
  g,2 c4 d8 e8 
  f4 c4 cis4 d4   %10
  e1 
  e8 d8 c2 b8 a8 
  b4 e2 d4 
  e4 f4 c2 
  a4 d4 g,8 g'4 f8~   %15
  f8 e8 f8 fis8 g8 b,16 c16 d8 dis8 
  e2 e8 c8 g'8 f8 
  e4. d8 c8 b8 a8 b8 
  c2 a8 b8 c8 d8 
  e8 b8 c8 d8 g,4 d'8 e8   %20
  f8 a,8 b8 cis8 d8 f,8 g8 a8 
  d,2 d'8 e8 f8 d8 
  bes8 b8 c4. cis8 d8 dis8 
  e8 fis8 g8 e8 c8 a8 d4 
  g,2 c8 bes8 a8 g8   %25
  f8 a8 bes8 b8 c8 cis8 d4~ 
  d8 c16 b16 c8 d8 e2 
  e,4 e'2 d8 c8 
  g'2 r4 g,8 a8 
  b4 e,8 fis8 g8 b8 c8 d8   %30
  g,2 e8 e'8 d8 c8 
  b8 b'4 a8 g8 fis8 e4~ 
  e8 b8 c8 d8 g,4 d'8 e8 
  f8 fis8 g4. gis8 a8 a,8 
  d2 g,8 g'4 fis16 e16   %35
  fis8 b8 e,4. dis16 cis16 dis8 b8 
  e8 fis8 g8 e8 c8 a8 d4 
  g,2 g'8 cis,8 d4~ 
  d8 c8 b8 bes8 a4. gis8 
  a8 b8 c8 d8 e2   %40
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