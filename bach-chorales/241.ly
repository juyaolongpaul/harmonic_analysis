\version "2.10.33"

\header {
  title = "241 - Was willst du dich, o meine Seele"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \minor
}


soprano = \relative c'' {
  \partial 4 a4 
  d,4. e8 f4 g4 
  a8 b8 c4 b4 a4 
  a4 gis4 a4 a4 
  g4 g4 c4 a4 
  bes8 a8 g8 f8 g4. f8   %5
  f2 r4 a4 
  d,4 e4 f4 g4 
  a8 b8 c4 b4. a8 
  a2 r4 a4 
  g4 g4 c4 a4   %10
  bes8 a8 g8 f8 e4. d8 
  d2 r4 a'4 
  g8 a8 f8 g8 e4. d8 
  d2 r4 a'4 
  a4 b4 c4 c4   %15
  b2 a4 a4 
  g4 g4 c4 a4 
  bes8 a8 g8 f8 g4. f8 
  f2 r4 a4 
  d,4 e4 f8 g8 a4   %20
  d,2 r4 e4 
  f4 e4 f4 g4 
  a4 b4 c8 b8 a4 
  d4 c4 bes8 a8 g8 f8 
  e4. d8 d2   %25
  
}

alto = \relative c' {
  \partial 4 a4 
  a4 b8 cis8 d8 f4 e8 
  f8 g8 g8 a8 g8 f8 e4 
  f4 e4 e4 e8 f8~ 
  f8 e16 d16 e8 f8 g8 e8 f4 
  f4 e8 f4 e16 d16 e4   %5
  c2 r4 d4~ 
  d8 a8 bes8 a4 d4 cis8 
  d4 g8 f4 d8 b8 e16 d16 
  c2 r4 f4~ 
  f8 e16 d16 e8 f8 g8 e8 f8 a8   %10
  g8 f8 e8 d8 d8 cis16 b16 cis4 
  a2 r4 f'4 
  e4 d4. cis16 b16 cis4 
  d2 r4 c4 
  c8 d16 e16 f8 e16 d16 g8 f16 g16 a4~   %15
  a8 gis16 fis16 gis4 e4 f4 
  f4 f8 e16 d16 e4 f4 
  f16 e16 f8 e8 f4 e16 d16 e4 
  c2 r4 c4 
  b4 cis4 d4 cis4   %20
  d2 r4 cis4 
  d4 cis4 d8 c8 bes8 c8 
  c8 d16 e16 f8 e16 d16 e8 f16 g16 a8 g8 
  fis8 g8 a8 fis8 d4 e8 d8~ 
  d8 cis16 b16 cis4 a2   %25
  
}

tenor = \relative c {
  \partial 4 f8  g8 
  a4 g4 a4 c4 
  c8 d8 e4 d4. c8 
  b4. d8 c4 c4 
  d8 g,8 c4 c4 c4 
  d8 a8 c4 d8 bes8 g8 c16 bes16   %5
  a2 r4 f8 g8 
  a4 g4 f8 bes4. 
  a8 g8 g8 a4 gis16 fis16 g4 
  e2 r4 c'4 
  d8 g,8 c4 c4 c8 f8   %10
  d8 cis4 d8 bes8 g4 a16 g16 
  f2 r4 d'8 c8 
  bes8 a8 a8 g16 a16 bes8 g8 e8 a16 g16 
  f2 r4 f8 g8 
  a8 b16 c16 d8 b8 g8 c16 b16 a8 c8   %15
  f8 d8 b8 e16 d16 cis4 d4~ 
  d8 c16 b16 c4 c4. d16 c16 
  bes8 c8 c16 bes16 a8 g16 a16 bes4 c16 bes16 
  a2 r4 f4 
  g4. f16 g16 a4. g8   %20
  f2 r4 a4 
  a4 a4 a4 d8 c16 bes16 
  a8 b16 c16 d8 b8 g8 d'8 c4~ 
  c8 bes8 a8 d4 c8 bes8 b8 
  e,4~ e16 f16 g8 fis2   %25
  
}

baixo = \relative c {
  \partial 4 d8  e8 
  f8 g8 f8 e8 d4 c4 
  f4 e8 fis8 g8 gis8 a8 c,8 
  d8 b8 e4 a,4 a4 
  b4 c8 d8 e8 c8 f8 ees8 
  d8 c8 bes8 a8 bes8 g8 c4   %5
  f,2 r4 d'8 e8 
  f4 cis4 d4 e4 
  f8 g16 f16 e8 f16 e16 d8 b8 e4 
  a,2 r4 a4 
  b4 c8 d8 e8 c8 f8 d8   %10
  g8 a8 bes8 a8 g8 e8 a8 a,8 
  d2 r4 d4~ 
  d8 cis8 d8 bes8 g8 e8 a4 
  bes2 r4 f'4 
  f8 e8 d8 g8 e8 a16 g16 f8 e8   %15
  d8 b8 e4 a,4 d8 c8 
  b8 g8 c8 bes8 a8 g8 f4 
  g8 a16 bes16 c8 d8 bes8 g8 c8 c,8 
  f2 r4 f'4~ 
  f4 e4 d4 a4   %20
  bes2 r4 a4 
  d8 f8 a8 g8 f4. e8 
  f8 e8 d8 g8 c,8 d16 e16 f8 e8 
  d8 e8 fis8 d8 g4. gis8 
  a4 a,4 d2   %25
  
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