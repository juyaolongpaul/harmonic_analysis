\version "2.10.33"

\header {
  title = "306 - O Mensch, bewein' dein' Sünde groß"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key ees \major
}


soprano = \relative c' {
  \repeat volta 2 {
    \partial 4 ees4 
    ees4 f4 g8 f8 ees4 
    g4 a4 bes4 bes4 
    aes4 g4 f4 ees8 f8 
    g4 a4 bes4 c4 
    bes4 aes4 g4 aes4     %5
    f2 ees4 } ees'4 
  ees4 ees4 d4 bes4 
  c4 c4 g4 ees'4 
  ees4 ees4 d4 bes4 
  c4 c4 bes4 g8 aes8   %10
  bes4 c4 bes4 g4 
  aes2 g4 f4 
  g4 aes4 bes4 bes4 
  aes4 g4 f4 bes4 
  c4 d4 ees4 d4   %15
  c4 c4 bes4 ees,4 
  f4 g4 aes4 g4 
  f2 ees4 
  
}

alto = \relative c' {
  \repeat volta 2 {
    \partial 4 bes4 
    c4. b16 c16 d4. c16 d16 
    ees4. d8 d4 g4~ 
    g8 f4 ees4 d8 c8 d8 
    ees4 ees4 d4 f4~ 
    f8 ees16 d16 ees8 bes8 bes ees16 d16 ees4~     %5
    ees8 d16 c16 d4 bes4 } bes'8 aes8 
  g4 f4 f4 ees4 
  ees8 des8 ees4 ees4 g4 
  f4 f4 f4 f4 
  g4 f8 ees8 d4 ees8 f8   %10
  g4 aes4 g4 g4~ 
  g8 f16 ees16 f8 d8 ees4 f4 
  ees4 ees4 f4 g4 
  c,8 d8 ees4. d8 f4 
  f4 f4 g8 a8 bes4~   %15
  bes4 a4 g4 ees4 
  d4 ees4 f8 d8 bes8 ees8~ 
  ees8 d16 c16 d4 bes4 
  
}

tenor = \relative c' {
  \repeat volta 2 {
    \partial 4 g4 
    aes4 aes4 g4 g4 
    c8 bes8 a4 g4 d'4 
    ees8 bes8 bes4 c8 g8 g4 
    c8 bes8 c4 bes4 aes4~ 
    aes8 g4 f8 ees4. bes'8     %5
    c8 aes8 f8 bes8 g4 } g8 aes8 
  bes4 c4. bes16 aes16 g4 
  aes4 aes4 bes4 c4
  c8 bes8 a4 bes8 a8 bes4~ 
  bes8 a16 g16 a4 bes4 bes4   %10
  ees4 ees4 ees8 des4 c8 
  c4 d8 f8 c4 aes4 
  bes4 c4 d4 d,8 e8 
  f4 bes4 bes4 d4 
  c4 bes4. c8 d4   %15
  g,4 d'4 d4 c4 
  bes4 bes4. f8 g8 bes8 
  c8 aes8 f8 bes16 aes16 g4 
  
}

baixo = \relative c {
  \repeat volta 2 {
    \partial 4 ees4 
    aes8 g8 f4 b,4 c4~ 
    c4 fis,4 g4 g'4 
    c,8 d8 ees4 a,8 b8 c4~ 
    c4 f,4 bes4 f'8 ees8 
    d8 bes8 c8 d8 ees8 d8 c8 bes8     %5
    aes8 f8 bes4 ees,4 } ees8 f8 
  g4 a4 bes8 c8 des4 
  c8 bes8 aes4 ees'4 c8 bes8 
  a8 g8 f4 bes8 c8 d4 
  c4 f4 bes,4 ees4~   %10
  ees8 des8 c8 d8 ees4 e4 
  f4 b,4 c4 des4~ 
  des4 c4 bes8 aes8 g4 
  f4 g8 aes8 bes4 bes'4 
  a4 aes4 g4. f8   %15
  e4 fis4 g4 aes4~ 
  aes8 g8 f8 ees8 d8 bes8 ees8 g,8 
  aes4 bes4 ees,4 
  
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